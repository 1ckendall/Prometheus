from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, title="", xlabel="", ylabel=""):
        fig = Figure(figsize=(4, 3), dpi=100)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

        self.title_text = title
        self.xlabel_text = xlabel
        self.ylabel_text = ylabel

        self.axes.set_title(title, color="white")
        self.axes.set_xlabel(xlabel, color="white")
        self.axes.set_ylabel(ylabel, color="white")
        self.axes.tick_params(colors="white")
        self.axes.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        self.draw()
