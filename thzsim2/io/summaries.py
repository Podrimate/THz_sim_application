from pathlib import Path
import csv

from thzsim2.models.reference import ReferenceSummary

SUMMARY_FIELDNAMES = ("parameter", "value", "unit")


def write_reference_summary_csv(path, summary: ReferenceSummary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for parameter, value, unit in summary.as_rows():
            writer.writerow(
                {
                    "parameter": parameter,
                    "value": format(float(value), ".16g") if isinstance(value, float) else str(value),
                    "unit": unit,
                }
            )


def render_reference_summary_text(summary: ReferenceSummary):
    rows = [(parameter, str(value), unit) for parameter, value, unit in summary.as_rows()]
    p_width = max(len("parameter"), max(len(parameter) for parameter, _, _ in rows))
    v_width = max(len("value"), max(len(value) for _, value, _ in rows))
    lines = [
        f"{'parameter':<{p_width}}  {'value':<{v_width}}  unit",
        f"{'-' * p_width}  {'-' * v_width}  ----",
    ]
    for parameter, value, unit in rows:
        lines.append(f"{parameter:<{p_width}}  {value:<{v_width}}  {unit}")
    return "\n".join(lines) + "\n"


def write_reference_summary_txt(path, summary: ReferenceSummary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_reference_summary_text(summary), encoding="utf-8")


def render_sample_structure_text(*, freq_grid_thz, n_in, n_out, layers):
    lines = [
        "Sample Structure",
        "",
        f"Ambient media: n_in = {float(n_in):.16g}, n_out = {float(n_out):.16g}",
        (
            "Frequency grid: "
            f"{len(freq_grid_thz)} points from {float(freq_grid_thz[0]):.16g} THz "
            f"to {float(freq_grid_thz[-1]):.16g} THz"
        ),
        "",
    ]

    for layer in layers:
        lines.append(f"Layer {layer['index'] + 1}: {layer['name']}")
        thickness = f"{layer['thickness_um']:.16g} um"
        if layer["thickness_fit"] is not None:
            thickness += (
                f"  [fit: {layer['thickness_fit']['bound_min']:.16g}"
                f" to {layer['thickness_fit']['bound_max']:.16g} um]"
            )
        lines.append(f"  thickness_um = {thickness}")
        lines.append(f"  material = {layer['material_kind']}")
        if layer.get("source_nk_file"):
            lines.append(f"  source_nk_file = {layer['source_nk_file']}")
        for parameter in layer["parameters"]:
            value_text = f"{parameter['value']:.16g} {parameter['unit']}".strip()
            if parameter.get("fit") is not None:
                fit = parameter["fit"]
                value_text += f"  [fit: {fit['bound_min']:.16g} to {fit['bound_max']:.16g} {parameter['unit']}]"
            lines.append(f"  {parameter['name']} = {value_text}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_sample_structure_txt(path, *, freq_grid_thz, n_in, n_out, layers):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_sample_structure_text(freq_grid_thz=freq_grid_thz, n_in=n_in, n_out=n_out, layers=layers),
        encoding="utf-8",
    )
