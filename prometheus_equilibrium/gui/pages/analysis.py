import math

import numpy as np
from PySide6.QtWidgets import QGridLayout, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from prometheus_equilibrium.gui.widgets.graph_canvas import GraphCanvas

_NO_HISTORY_MSG = (
    'Enable "Record Convergence History"\n'
    "in the Solver Options panel to view\n"
    "convergence plots."
)

_DEFAULT_RESULTS_REPORT_TEXT = (
    "=== Rocket Performance Report ===\n\n"
    "Run a calculation to populate shared chamber state,\n"
    "frozen/shifting exit comparison, and performance tables."
)


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
        self.results_text.setText(_DEFAULT_RESULTS_REPORT_TEXT)
        layout.addWidget(self.results_text)

    def reset_results_view(self) -> None:
        """Clear report text and plots after composition or input changes."""
        self.results_text.setText(_DEFAULT_RESULTS_REPORT_TEXT)
        placeholder = "Results cleared.\nRun a new calculation."
        canvases = (
            self.canvas_concentration,
            self.canvas_temperature_conv,
            self.canvas_exp_temp,
            self.canvas_exp_mach,
            self.canvas_exp_mw,
            self.canvas_exp_gamma,
            self.canvas_isp,
            self.canvas_cstar,
            self.canvas_tc,
            self.canvas_expansion,
        )
        for canvas in canvases:
            self._placeholder(canvas, placeholder)

    # ------------------------------------------------------------------ helpers

    def _setup_axes(self, canvas, title=None, xlabel=None, ylabel=None):
        """Clear a canvas and re-apply dark-theme axis styling."""
        canvas.axes.clear()
        canvas.axes.set_title(title or canvas.title_text, color="white")
        canvas.axes.set_xlabel(
            xlabel if xlabel is not None else canvas.xlabel_text, color="white"
        )
        canvas.axes.set_ylabel(
            ylabel if ylabel is not None else canvas.ylabel_text, color="white"
        )
        canvas.axes.tick_params(colors="white")
        canvas.axes.grid(True, linestyle="--", alpha=0.3)

    def _placeholder(self, canvas, message):
        """Show a centered placeholder message on a canvas."""
        self._setup_axes(canvas, xlabel="", ylabel="")
        canvas.axes.tick_params(colors="white", labelbottom=False, labelleft=False)
        canvas.axes.grid(False)
        canvas.axes.text(
            0.5,
            0.5,
            message,
            transform=canvas.axes.transAxes,
            ha="center",
            va="center",
            color="#aaaaaa",
            fontsize=10,
            style="italic",
            wrap=True,
        )
        canvas.draw()

    # ------------------------------------------------------- convergence plots

    def update_convergence_plots(self, solution):
        """Update the convergence tab with data from the last solve."""
        if not solution.history:
            self._placeholder(self.canvas_concentration, _NO_HISTORY_MSG)
            self._placeholder(self.canvas_temperature_conv, _NO_HISTORY_MSG)
            return

        iterations = list(range(1, len(solution.history) + 1))

        # 1. Temperature plot
        temps = [step.temperature for step in solution.history]
        self._setup_axes(self.canvas_temperature_conv)
        self.canvas_temperature_conv.axes.plot(iterations, temps, "o-", color="#2a82da")
        self.canvas_temperature_conv.draw()

        # 2. Concentration plot (top 10 species)
        self._setup_axes(self.canvas_concentration)

        all_sp_names = set()
        for step in solution.history:
            all_sp_names.update(step.mole_fractions.keys())

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

    # -------------------------------------------------------- expansion plots

    def update_expansion_plots(self, result):
        """Update expansion plots from a RocketPerformanceComparison.

        Always shows at least chamber, throat, and exit as labelled points.
        When a full nozzle profile is available it draws the smooth curve
        instead.
        """
        shifting = getattr(result, "shifting", result)
        frozen = getattr(result, "frozen", None)

        _STATION_LABELS = ["Chamber", "Throat", "Exit"]

        def _mach(perf, sol):
            if hasattr(sol, "mach_number"):
                return sol.mach_number
            dH_mass = perf.chamber.total_enthalpy - sol.total_enthalpy
            v = math.sqrt(max(0.0, 2 * dH_mass))
            return v / sol.speed_of_sound

        def _extract(perf):
            if perf.profile:
                P = [sol.pressure for sol in perf.profile]
                T = [sol.temperature for sol in perf.profile]
                MW = [sol.gas_mean_molar_mass * 1000 for sol in perf.profile]
                gamma_vals = [sol.gamma for sol in perf.profile]
                mach_vals = [_mach(perf, sol) for sol in perf.profile]
                return P, T, mach_vals, MW, gamma_vals, True
            # Minimum: chamber + throat + exit
            stations = [perf.chamber, perf.throat, perf.exit]
            P = [s.pressure for s in stations]
            T = [s.temperature for s in stations]
            MW = [s.gas_mean_molar_mass * 1000 for s in stations]
            gamma_vals = [s.gamma for s in stations]
            mach_vals = [_mach(perf, s) for s in stations]
            return P, T, mach_vals, MW, gamma_vals, False

        sP, sT, sMach, sMW, sGamma, s_profile = _extract(shifting)
        fP = fT = fMach = fMW = fGamma = None
        if frozen is not None:
            fP, fT, fMach, fMW, fGamma, _ = _extract(frozen)

        def _plot(canvas, y_s, y_f_or_none, ylabel):
            self._setup_axes(canvas, ylabel=ylabel)
            ax = canvas.axes

            style_s = "o-" if s_profile else "o--"
            ms = 4 if s_profile else 7
            ax.semilogx(
                sP, y_s, style_s, markersize=ms, color="#2a82da", label="Shifting"
            )

            if not s_profile:
                for xi, yi, lbl in zip(sP, y_s, _STATION_LABELS):
                    ax.annotate(
                        lbl,
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        color="#2a82da",
                        fontsize=7,
                    )

            if y_f_or_none is not None and fP is not None:
                style_f = "s--" if s_profile else "s:"
                ax.semilogx(
                    fP,
                    y_f_or_none,
                    style_f,
                    markersize=ms,
                    color="#f39c12",
                    label="Frozen",
                )
                if not s_profile:
                    for xi, yi, lbl in zip(fP, y_f_or_none, _STATION_LABELS):
                        ax.annotate(
                            lbl,
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(0, -13),
                            ha="center",
                            color="#f39c12",
                            fontsize=7,
                        )
                ax.legend(fontsize="small", framealpha=0.5)

            ax.invert_xaxis()
            canvas.draw()

        _plot(self.canvas_exp_temp, sT, fT, "T (K)")
        _plot(self.canvas_exp_mach, sMach, fMach, "Mach")
        _plot(self.canvas_exp_mw, sMW, fMW, "g/mol")
        _plot(self.canvas_exp_gamma, sGamma, fGamma, "gamma")

    # -------------------------------------------------- performance viz plots

    def update_performance_plots(
        self, cases, sweep_axis="none", sweep_label="Run Index"
    ):
        """Update performance charts from O/F or Pc sweep results.

        For single-point (non-sweep) runs, shows a per-station summary of key
        properties instead of leaving the tab empty.
        """
        sweep_cases = [(x, perf) for x, perf in cases if x is not None]
        if not sweep_cases:
            _, perf = cases[0]
            self._update_single_point_plots(perf)
            return

        x = [v for v, _ in sweep_cases]
        isp_shift = [perf.shifting.isp_actual for _, perf in sweep_cases]
        isp_frozen = [perf.frozen.isp_actual for _, perf in sweep_cases]
        cstar_shift = [perf.shifting.cstar for _, perf in sweep_cases]
        cstar_frozen = [perf.frozen.cstar for _, perf in sweep_cases]
        tc = [perf.shifting.chamber.temperature for _, perf in sweep_cases]
        area_shift = [perf.shifting.area_ratio for _, perf in sweep_cases]
        area_frozen = [perf.frozen.area_ratio for _, perf in sweep_cases]

        def _line(canvas, y1, y2=None, label1="Shifting", label2="Frozen"):
            self._setup_axes(canvas, xlabel=sweep_label)
            canvas.axes.plot(x, y1, "o-", color="#2a82da", label=label1)
            if y2 is not None:
                canvas.axes.plot(x, y2, "s--", color="#f39c12", label=label2)
                canvas.axes.legend(fontsize="small", framealpha=0.5)
            canvas.axes.tick_params(axis="x", labelrotation=30)
            canvas.figure.subplots_adjust(bottom=0.22)
            canvas.draw()

        _line(self.canvas_isp, isp_shift, isp_frozen)
        _line(self.canvas_cstar, cstar_shift, cstar_frozen)
        _line(self.canvas_tc, tc, label1="Chamber")
        _line(
            self.canvas_expansion,
            area_shift,
            area_frozen,
            label1="Ae/At Shift",
            label2="Ae/At Frozen",
        )

    def _update_single_point_plots(self, perf):
        """Performance summary charts for a single (non-sweep) calculation.

        Repurposes the four performance canvases to show a station-by-station
        breakdown of the key properties rather than leaving them blank.
        """
        shifting = perf.shifting
        frozen = perf.frozen
        stations = ["Chamber", "Throat", "Exit"]
        x = np.arange(3)
        width = 0.35

        # ---- Isp comparison: actual / vac / SL for shifting vs frozen ----
        categories = ["Actual", "Vacuum", "Sea Level"]
        isp_s = [shifting.isp_actual, shifting.isp_vac, shifting.isp_sl]
        isp_f = [frozen.isp_actual, frozen.isp_vac, frozen.isp_sl]
        xc = np.arange(len(categories))

        self._setup_axes(
            self.canvas_isp, title="Specific Impulse", xlabel="", ylabel="Isp (s)"
        )
        ax = self.canvas_isp.axes
        bars_s = ax.bar(xc - width / 2, isp_s, width, label="Shifting", color="#2a82da")
        bars_f = ax.bar(xc + width / 2, isp_f, width, label="Frozen", color="#f39c12")
        ax.set_xticks(xc)
        ax.set_xticklabels(categories)
        ax.tick_params(colors="white")
        ax.legend(fontsize="small", framealpha=0.5)
        for bar in list(bars_s) + list(bars_f):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=7,
            )
        self.canvas_isp.draw()

        # ---- C* comparison: shifting vs frozen ----
        self._setup_axes(
            self.canvas_cstar,
            title="Characteristic Velocity (C*)",
            xlabel="",
            ylabel="C* (m/s)",
        )
        ax = self.canvas_cstar.axes
        cstar_vals = [shifting.cstar, frozen.cstar]
        bars = ax.bar(
            ["Shifting", "Frozen"],
            cstar_vals,
            color=["#2a82da", "#f39c12"],
            width=0.4,
        )
        ax.tick_params(colors="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=9,
            )
        self.canvas_cstar.draw()

        # ---- Station temperatures: chamber / throat / exit ----
        T_s = [
            shifting.chamber.temperature,
            shifting.throat.temperature,
            shifting.exit.temperature,
        ]
        T_f = [
            frozen.chamber.temperature,
            frozen.throat.temperature,
            frozen.exit.temperature,
        ]
        self._setup_axes(
            self.canvas_tc, title="Station Temperature", xlabel="", ylabel="T (K)"
        )
        ax = self.canvas_tc.axes
        ax.bar(x - width / 2, T_s, width, label="Shifting", color="#2a82da")
        ax.bar(x + width / 2, T_f, width, label="Frozen", color="#f39c12")
        ax.set_xticks(x)
        ax.set_xticklabels(stations)
        ax.tick_params(colors="white")
        ax.legend(fontsize="small", framealpha=0.5)
        self.canvas_tc.draw()

        # ---- Station gamma: chamber / throat / exit ----
        g_s = [shifting.chamber.gamma, shifting.throat.gamma, shifting.exit.gamma]
        g_f = [frozen.chamber.gamma, frozen.throat.gamma, frozen.exit.gamma]
        self._setup_axes(
            self.canvas_expansion,
            title="Station Gamma (\u03b3)",
            xlabel="",
            ylabel="\u03b3",
        )
        ax = self.canvas_expansion.axes
        ax.bar(x - width / 2, g_s, width, label="Shifting", color="#2a82da")
        ax.bar(x + width / 2, g_f, width, label="Frozen", color="#f39c12")
        ax.set_xticks(x)
        ax.set_xticklabels(stations)
        ax.tick_params(colors="white")
        ax.legend(fontsize="small", framealpha=0.5)
        # Tight y-range so small gamma differences are visible
        all_g = g_s + g_f
        g_min, g_max = min(all_g), max(all_g)
        margin = max(0.02, (g_max - g_min) * 0.3)
        ax.set_ylim(g_min - margin, g_max + margin)
        self.canvas_expansion.draw()
