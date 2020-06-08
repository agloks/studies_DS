from importlib.machinery import SourceFileLoader
import os

file_directory = os.path.realpath(__file__)
file_directory_splited = file_directory.split("/")
file_directory_without_file = "/".join(file_directory_splited[:-1])


Graph = SourceFileLoader("Graph", f"{file_directory_without_file}/graph.py").load_module()
Refine = SourceFileLoader("Refine", f"{file_directory_without_file}/refine.py").load_module()
AlgoML = SourceFileLoader("AlgoML", f"{file_directory_without_file}/ml.py").load_module()
Handler = SourceFileLoader("Handler", f"{file_directory_without_file}/handler.py").load_module()
Metrics = SourceFileLoader("Metrics", f"{file_directory_without_file}/metrics.py").load_module()