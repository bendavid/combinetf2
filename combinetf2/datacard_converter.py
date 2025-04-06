import os

import hist
import numpy as np
from tqdm import tqdm
from wums import logging

from combinetf2 import tensorwriter
from combinetf2.datacard_parser import DatacardParser

logger = logging.child_logger(__name__)


class DatacardConverter:
    """
    Convert data from Combine datacards and ROOT files to hdf5 tensor format
    """

    def __init__(self, datacard_file, symmetrize=None, use_root=False, mass="125.38"):
        """
        Initialize the converter with a datacard file

        Args:
            datacard_file: Path to the datacard file
        """
        self.datacard_file = datacard_file
        self.parser = DatacardParser()
        self.mass = mass
        self.symmetrize = symmetrize

        # For counting experiments
        self.yield_axis = hist.axis.Integer(
            0, 1, name="yield", overflow=False, underflow=False
        )

        self.use_root = use_root
        if self.use_root:
            import ROOT
            from narf.histutils import root_to_hist

            self.io = ROOT
            self.root_to_hist = root_to_hist
            self.root_files = (
                {}
            )  # Cache for opened ROOT files and their directories as uproot objects
        else:
            import uproot

            self.io = uproot
            self.root_directories = {}  # Cache for opened ROOT files

    def parse(self):
        """Parse the datacard file"""
        self.parser.parse_file(self.datacard_file)
        return self

    def get_root_file(self, file_path):
        """Get a ROOT file, opening it if not already open"""
        if file_path not in self.root_files:
            # Check if path is relative to the datacard location
            datacard_dir = os.path.dirname(os.path.abspath(self.datacard_file))
            if not os.path.isabs(file_path):
                file_path = os.path.join(datacard_dir, file_path)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ROOT file not found: {file_path}")

            self.root_files[file_path] = self.io.TFile.Open(file_path, "READ")

        return self.root_files[file_path]

    def load_root_directories(self, file_path):
        """Get a ROOT file, opening it if not already open"""
        if file_path not in self.root_directories:
            # Check if path is relative to the datacard location
            datacard_dir = os.path.dirname(os.path.abspath(self.datacard_file))
            if not os.path.isabs(file_path):
                file_path = os.path.join(datacard_dir, file_path)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ROOT file not found: {file_path}")

            logger.info(f"Open file {file_path}")

            file = self.io.open(file_path)

            search_depth = self.parser.max_depth - 1
            if search_depth >= 0:
                # recursively find all directories up to a depth of 'search_depth', setting 'search_depth' to a proper value speeds up things (if you know how deep the directories go)
                def get_directories(directory, level=0):
                    keys = set([k.split("/")[0] for k in directory.keys()])
                    directories = [
                        d
                        for d in keys
                        if isinstance(directory[d], self.io.reading.ReadOnlyDirectory)
                    ]
                    if len(directories) == 0:
                        return directory
                    elif level >= search_depth:
                        return {d: directory[d] for d in directories}
                    else:
                        return {
                            d: get_directories(directory[d], level=level + 1)
                            for d in directories
                        }

                self.root_directories[file_path] = get_directories(file)
            else:
                self.root_directories[file_path] = file

    def is_TH1(self, histogram):
        if self.use_root:
            return isinstance(histogram, self.io.TH1)
        else:
            return histogram.classname.startswith("TH1")

    def get_histogram(
        self,
        shape_info,
        process,
        bin_name,
        systematic=None,
        demand=True,
        edges=False,
        variances=None,
    ):
        """
        Get a histogram based on shape information

        Args:
            shape_info: Shape directive information
            process: Process name
            bin_name: Bin/channel name
            systematic: Systematic uncertainty (None for nominal)
            demand: throw an error if shape is not found and demand=True, otherwise not

        Returns:
            TH1 histogram or None if not found
        """
        if not shape_info:
            return None

        file_path = shape_info["file"]

        if systematic and "histogram_syst_pattern" in shape_info:
            syst_pattern = shape_info["histogram_syst_pattern"]
            hist_name = syst_pattern.replace("$PROCESS", process)
            hist_name = hist_name.replace("$CHANNEL", bin_name)
            hist_name = hist_name.replace("$SYSTEMATIC", systematic)
            hist_name = hist_name.replace("$MASS", self.mass)
        else:
            # Replace variables in histogram pattern
            hist_pattern = shape_info["histogram_pattern"]
            hist_name = hist_pattern.replace("$PROCESS", process)
            hist_name = hist_name.replace("$CHANNEL", bin_name)
            hist_name = hist_name.replace("$MASS", self.mass)

            if systematic:
                # If no separate systematic pattern, assume appending Up/Down to nominal
                hist_name = f"{hist_name}_{systematic}"

        if self.use_root:
            root_file = self.get_root_file(file_path)
            histogram = root_file.Get(hist_name)
            if hasattr(histogram, "SetDirectory"):
                histogram.SetDirectory(0)
        else:
            # Try to find the histogram in the file
            histogram = self.root_directories[file_path]
            for p in hist_name.split("/"):
                histogram = histogram.get(p, {})

        if not histogram or not self.is_TH1(histogram):
            if demand:
                raise ValueError(f"Histogram {hist_name} not found in {file_path}")
            else:
                logger.debug(
                    f"Histogram {hist_name} not found in {file_path}, but 'demand=False'. Skip it"
                )
                return None
        else:
            if self.use_root:
                h = self.root_to_hist(histogram)
                histogram.Delete()
                return h
            else:
                return histogram.to_hist()

    def convert_to_hdf5(self, sparse=False):
        """
        Convert the datacard and histograms to numpy arrays

        Returns:
            Dictionary of numpy arrays with all the data
        """
        logger.info("Parse datacard text")
        self.parse()
        logger.info("Prepare histograms")
        # Get shape directives for each process and bin
        shape_map = {}
        for shape in self.parser.shapes:
            process = shape["process"]
            channel = shape["channel"]

            # Handle wildcards
            if process == "*":
                processes = ["data_obs", *self.parser.processes]
            else:
                processes = [process]

            if channel == "*":
                channels = self.parser.bins
            else:
                channels = [channel]

            for p in processes:
                for c in channels:
                    shape_map[(p, c)] = shape

            if not self.use_root:
                self.load_root_directories(shape["file"])

        logger.info("Convert histograms into hdf5 tensor")

        # TODO: nuisance groups

        writer = tensorwriter.TensorWriter(
            sparse=sparse,
        )

        logger.info("loop over channels (aka combine bins) for nominal histograms")
        for bin_name in tqdm(self.parser.bins, desc="Processing"):

            for process_name in ["data_obs", *self.parser.bin_process_map[bin_name]]:

                shape_info = (
                    shape_map.get((process_name, bin_name))
                    or shape_map.get(("*", bin_name))
                    or shape_map.get((process_name, "*"))
                    or shape_map.get(("*", "*"))
                )

                if shape_info:
                    h_proc = self.get_histogram(
                        shape_info,
                        process_name,
                        bin_name,
                    )
                else:
                    # For counting experiments
                    h_proc = hist.Hist(
                        self.yield_axis,
                        data=np.array([self.parser.observations.get(bin_name, 0)]),
                        storage=hist.storage.Double(),
                    )

                if process_name == "data_obs":
                    writer.add_channel(h_proc.axes, bin_name)
                    writer.add_data(h_proc, bin_name)
                else:
                    writer.add_process(
                        h_proc,
                        process_name,
                        bin_name,
                        signal=self.parser.process_indices.get(process_name, 0) <= 0,
                    )

        # Add rate parameters as unconstrained lnN uncertainties with 1% variation.
        #   To compare with combine use 'r = exp(100 * r)'
        for parts in self.parser.rate_params:
            name = parts[0]
            channels = [parts[1]] if parts[1] != "*" else self.parser.bins
            processes = [parts[2]] if parts[2] != "*" else self.parser.processes
            # TODO: set initial value
            value = float(parts[3])
            # TODO: use parameter range
            if len(parts) >= 5:
                lo, hi = parts[4][1:-1].split(",")
                prange = (float(lo), float(hi))

            for c in channels:
                writer.add_lnN_systematic(name, processes, c, 1.01, constrained=False)

        def add_lnN_syst(writer, name, process, channel, effect):
            # Parse the effect (could be asymmetric like 0.9/1.1)
            if "/" in effect:
                down, up = effect.split("/")
                writer.add_lnN_systematic(
                    name, process, channel, [(float(up), float(down))]
                )
            else:
                writer.add_lnN_systematic(name, process, channel, float(effect))

        logger.info("loop over systematic variations")
        for syst in tqdm(self.parser.systematics, desc="Processing"):

            if syst["type"] in ["shape", "shapeN", "shape?", "shapeU"]:
                # TODO dedicated treatment for shapeN
                for (bin_name, process_name), effect in syst["effects"].items():
                    if effect not in ["-", "0"]:

                        shape_info = (
                            shape_map.get((process_name, bin_name))
                            or shape_map.get(("*", bin_name))
                            or shape_map.get((process_name, "*"))
                            or shape_map.get(("*", "*"))
                        )

                        hist_up = self.get_histogram(
                            shape_info,
                            process_name,
                            bin_name,
                            f"{syst['name']}Up",
                            demand=syst["type"] != "shape?",
                        )
                        hist_down = self.get_histogram(
                            shape_info,
                            process_name,
                            bin_name,
                            f"{syst['name']}Down",
                            demand=syst["type"] != "shape?",
                        )

                        if hist_up is None and hist_down is None:
                            # 'syst?' case
                            add_lnN_syst(
                                writer, syst["name"], process_name, bin_name, effect
                            )
                        else:
                            writer.add_systematic(
                                [hist_up, hist_down],
                                syst["name"],
                                process_name,
                                bin_name,
                                kfactor=float(effect),
                                constrained=syst["type"]
                                != "shapeU",  # TODO check if shapeU is unconstriained
                                symmetrize=self.symmetrize,
                            )

            elif syst["type"] in ["lnN", "lnU"]:
                # TODO dedicated treatment for lnU

                for (bin_name, process_name), effect in syst["effects"].items():
                    if effect not in ["-", "0"]:
                        add_lnN_syst(
                            writer, syst["name"], process_name, bin_name, effect
                        )

        return writer

    def __del__(self):
        if self.use_root:
            logger.info("Close root files")
            for file in self.root_files.values():
                file.Close()
