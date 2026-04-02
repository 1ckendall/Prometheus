import math

from PySide6.QtWidgets import QGridLayout, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from prometheus_equilibrium.gui.widgets.graph_canvas import GraphCanvas


class AnalysisPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout(self)
        self.analysis_tabs = QTabWidget()

        self.tab_convergence = QWidget()
        self.tab_expansion = QWidget()
        self.tab_results_graphs = QWidget()
        self.tab_results_report = QWidget()

        self.analysis_tabs.addTab(self.tab_convergence, "Solver Convergence")
        self.analysis_tabs.addTab(self.tab_expansion, "Nozzle Expansion")
        self.analysis_tabs.addTab(self.tab_results_graphs, "Performance Visualization")
        self.analysis_tabs.addTab(self.tab_results_report, "Thermodynamic Report")

        self.setup_convergence_tab()
        self.setup_expansion_tab()
        self.setup_results_graphs_tab()
        self.setup_results_report_tab()

        layout.addWidget(self.analysis_tabs)

    def setup_convergence_tab(self):
        layout = QVBoxLayout(self.tab_convergence)
        self.canvas_concentration = GraphCanvas(
            self, "Log Concentration vs Iteration", "Iteration", "Log X"
        )
        self.canvas_temperature_conv = GraphCanvas(
            self, "Temperature Convergence", "Iteration", "K"
        )
        layout.addWidget(self.canvas_concentration)
        layout.addWidget(self.canvas_temperature_conv)

    def setup_expansion_tab(self):
        layout = QVBoxLayout(self.tab_expansion)
        grid = QGridLayout()

        self.canvas_exp_temp = GraphCanvas(
            self, "Temperature vs Pressure", "Pressure (Pa)", "T (K)"
        )
        self.canvas_exp_mach = GraphCanvas(
            self, "Mach Number vs Pressure", "Pressure (Pa)", "Mach"
        )
        self.canvas_exp_mw = GraphCanvas(
            self, "Mean Molar Mass vs Pressure", "Pressure (Pa)", "g/mol"
        )
        self.canvas_exp_gamma = GraphCanvas(
            self, "Gamma vs Pressure", "Pressure (Pa)", "gamma"
        )

        grid.addWidget(self.canvas_exp_temp, 0, 0)
        grid.addWidget(self.canvas_exp_mach, 0, 1)
        grid.addWidget(self.canvas_exp_mw, 1, 0)
        grid.addWidget(self.canvas_exp_gamma, 1, 1)

        layout.addLayout(grid)

    def setup_results_graphs_tab(self):
        layout = QVBoxLayout(self.tab_results_graphs)
        grid = QGridLayout()

        self.canvas_isp = GraphCanvas(self, "Specific Impulse", "Sweep", "s")
        self.canvas_cstar = GraphCanvas(self, "Characteristic Velocity", "Sweep", "m/s")
        self.canvas_tc = GraphCanvas(self, "Chamber Temperature", "Sweep", "K")
        self.canvas_expansion = GraphCanvas(
            self, "Expansion Properties", "Sweep", "Value"
        )

        grid.addWidget(self.canvas_isp, 0, 0)
        grid.addWidget(self.canvas_cstar, 0, 1)
        grid.addWidget(self.canvas_tc, 1, 0)
        grid.addWidget(self.canvas_expansion, 1, 1)

        layout.addLayout(grid)

    def setup_results_report_tab(self):
        layout = QVBoxLayout(self.tab_results_report)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFontFamily("Courier")
        self.results_text.setText(
            "=== Rocket Performance Report ===\n\n"
            "Run a calculation to populate shared chamber state,\n"
            "frozen/shifting exit comparison, and performance tables."
        )
        layout.addWidget(self.results_text)

    def update_convergence_plots(self, solution):
        """Update the convergence tab with data from the last solve."""
        if not solution.history:
            return

        iterations = list(range(1, len(solution.history) + 1))

        # 1. Temperature plot
        temps = [step.temperature for step in solution.history]
        self.canvas_temperature_conv.axes.clear()
        self.canvas_temperature_conv.axes.set_title(
            self.canvas_temperature_conv.title_text, color="white"
        )
        self.canvas_temperature_conv.axes.set_xlabel(
            self.canvas_temperature_conv.xlabel_text, color="white"
        )
        self.canvas_temperature_conv.axes.set_ylabel(
            self.canvas_temperature_conv.ylabel_text, color="white"
        )
        self.canvas_temperature_conv.axes.grid(True, linestyle="--", alpha=0.3)
        self.canvas_temperature_conv.axes.plot(iterations, temps, "o-", color="#2a82da")
        self.canvas_temperature_conv.draw()

        # 2. Concentration plot (top 10 species)
        import math

        self.canvas_concentration.axes.clear()
        self.canvas_concentration.axes.set_title(
            self.canvas_concentration.title_text, color="white"
        )
        self.canvas_concentration.axes.set_xlabel(
            self.canvas_concentration.xlabel_text, color="white"
        )
        self.canvas_concentration.axes.set_ylabel(
            self.canvas_concentration.ylabel_text, color="white"
        )
        self.canvas_concentration.axes.grid(True, linestyle="--", alpha=0.3)

        # Get all species that appeared in the history
        all_sp_names = set()
        for step in solution.history:
            all_sp_names.update(step.mole_fractions.keys())

        # Pick top 10 by final concentration
        final_step = solution.history[-1]
        sorted_sp = sorted(
            all_sp_names,
            key=lambda name: final_step.mole_fractions.get(name, 0.0),
            reverse=True,
        )
        top_sp = sorted_sp[:10]

        for name in top_sp:
            y = []
            for step in solution.history:
                val = step.mole_fractions.get(name, 1e-20)
                y.append(math.log10(max(val, 1e-20)))
            self.canvas_concentration.axes.plot(iterations, y, label=name)

        if top_sp:
            self.canvas_concentration.axes.legend(
                fontsize="x-small", loc="upper right", framealpha=0.5
            )

        self.canvas_concentration.draw()

    def update_expansion_plots(self, result):
        """Update expansion plots from one or two RocketPerformanceResult objects."""
        shifting = getattr(result, "shifting", result)
        frozen = getattr(result, "frozen", None)

        if not shifting.profile:
            return

        def _extract_profile(perf):
            P = [sol.pressure for sol in perf.profile]
            T = [sol.temperature for sol in perf.profile]
            MW = [sol.gas_mean_molar_mass * 1000 for sol in perf.profile]
            gamma_vals = [sol.gamma for sol in perf.profile]
            mach_vals = []
            for sol in perf.profile:
                if hasattr(sol, "mach_number"):
                    mach_vals.append(sol.mach_number)
                else:
                    dH_mass = perf.chamber.total_enthalpy - sol.total_enthalpy
                    v = math.sqrt(max(0.0, 2 * dH_mass))
                    mach_vals.append(v / sol.speed_of_sound)
            return P, T, mach_vals, MW, gamma_vals

        sP, sT, sMach, sMW, sGamma = _extract_profile(shifting)
        fP, fT, fMach, fMW, fGamma = (None, None, None, None, None)
        if frozen is not None and frozen.profile:
            fP, fT, fMach, fMW, fGamma = _extract_profile(frozen)

        def _plot_two(canvas, x1, y1, label1, x2=None, y2=None, label2=None):
            canvas.axes.clear()
            canvas.axes.set_title(canvas.title_text, color="white")
            canvas.axes.set_xlabel(canvas.xlabel_text, color="white")
            canvas.axes.set_ylabel(canvas.ylabel_text, color="white")
            canvas.axes.grid(True, linestyle="--", alpha=0.3)
            canvas.axes.semilogx(
                x1, y1, "o-", markersize=4, color="#2a82da", label=label1
            )
            if x2 is not None and y2 is not None:
                canvas.axes.semilogx(
                    x2, y2, "s--", markersize=4, color="#f39c12", label=label2
                )
                canvas.axes.legend(fontsize="small", framealpha=0.5)
            canvas.axes.invert_xaxis()  # Chamber (high P) on left
            canvas.draw()

        _plot_two(self.canvas_exp_temp, sP, sT, "Shifting", fP, fT, "Frozen")
        _plot_two(self.canvas_exp_mach, sP, sMach, "Shifting", fP, fMach, "Frozen")
        _plot_two(self.canvas_exp_mw, sP, sMW, "Shifting", fP, fMW, "Frozen")
        _plot_two(self.canvas_exp_gamma, sP, sGamma, "Shifting", fP, fGamma, "Frozen")

    def update_performance_plots(
        self, cases, sweep_axis="none", sweep_label="Run Index"
    ):
        """Update performance charts from O/F or Pc sweep results."""
        sweep_cases = [(x, perf) for x, perf in cases if x is not None]
        if not sweep_cases:
            return

        x = [of for of, _ in sweep_cases]
        isp_shift = [perf.shifting.isp_actual for _, perf in sweep_cases]
        isp_frozen = [perf.frozen.isp_actual for _, perf in sweep_cases]
        cstar_shift = [perf.shifting.cstar for _, perf in sweep_cases]
        cstar_frozen = [perf.frozen.cstar for _, perf in sweep_cases]
        tc = [perf.shifting.chamber.temperature for _, perf in sweep_cases]
        area_shift = [perf.shifting.area_ratio for _, perf in sweep_cases]
        area_frozen = [perf.frozen.area_ratio for _, perf in sweep_cases]

        def _plot(canvas, y1, y2=None, label1="Shifting", label2="Frozen"):
            canvas.axes.clear()
            canvas.axes.set_title(canvas.title_text, color="white")
            canvas.axes.set_xlabel(sweep_label, color="white")
            canvas.axes.set_ylabel(canvas.ylabel_text, color="white")
            canvas.axes.grid(True, linestyle="--", alpha=0.3)
            canvas.axes.plot(x, y1, "o-", color="#2a82da", label=label1)
            if y2 is not None:
                canvas.axes.plot(x, y2, "s--", color="#f39c12", label=label2)
                canvas.axes.legend(fontsize="small", framealpha=0.5)
            # Improve readability for dense sweeps and avoid clipped x-labels.
            canvas.axes.tick_params(axis="x", labelrotation=30)
            canvas.figure.subplots_adjust(bottom=0.22)
            canvas.draw()

        _plot(self.canvas_isp, isp_shift, isp_frozen)
        _plot(self.canvas_cstar, cstar_shift, cstar_frozen)
        _plot(self.canvas_tc, tc, None, label1="Chamber")
        _plot(
            self.canvas_expansion,
            area_shift,
            area_frozen,
            label1="Ae/At Shift",
            label2="Ae/At Frozen",
        )
