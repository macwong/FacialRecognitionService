class AlignOptions():
    def __init__(self, input_dir, output_dir, my_graph, detect_multiple_faces = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.my_graph = my_graph
        self.image_size = 160
        self.margin = 32
        self.random_order = True
        self.gpu_memory_fraction = 0.25
        self.detect_multiple_faces = detect_multiple_faces