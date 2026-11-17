
from pathlib import Path
import math
import textwrap
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.color import rgb2lab


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "processed" / "metadata.csv"

FIG = ROOT / "results" / "figures" / "phase4"
OUT = ROOT / "results" / "phase4"
REPORTS = ROOT / "reports"

QC = REPORTS / "image_quality_review.csv"
RULES = REPORTS / "inclusion_rules.md"
OBS = REPORTS / "phase4_observations.md"

# Original images are used. Test images must never be opened.
DEVELOPMENT_ROLES = {"train", "validation"}

CONDITIONS = [
    ("dino", "off", "Dino-Lite / lamp off"),
    ("dino", "on", "Dino-Lite / lamp on"),
    ("phone", "off", "Samsung A31 / lamp off"),
    ("phone", "on", "Samsung A31 / lamp on"),
]

INSTANCES = [1, 2, 3]
CONTACT_LEVELS = [0.00, 0.15, 0.50, 1.00, 2.00]

INSTANCE_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
}

MANUAL_QC_COLUMNS = [
    "focus_ok",
    "droplet_visible",
    "boundary_distinguishable",
    "not_severely_cropped",
    "no_severe_obstruction_or_glare",
    "usable",
    "exclusion_reason",
    "review_notes",
]


# ---------------------------------------------------------
# Metadata and image loading
# ---------------------------------------------------------

def image_path(value):
    """Resolve a metadata path safely inside the project."""
    p = (ROOT / str(value)).resolve()

    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError(
            f"Image path escapes project root: {value}"
        )

    if not p.is_file():
        raise FileNotFoundError(p)

    return p


def read_metadata():
    df = pd.read_csv(META)

    required = {
        "image_id",
        "image_path",
        "conc",
        "device",
        "lamp",
        "cohort",
        "instance_id",
        "tube_id",
        "role",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"metadata.csv is missing columns: {sorted(missing)}"
        )

    roles = set(df["role"].dropna().astype(str))

    if not DEVELOPMENT_ROLES.union({"test"}).issubset(roles):
        raise ValueError(
            f"Expected train, validation and test roles; found {roles}"
        )

    # Development = train + validation.
    # The locked test rows are removed before opening any image.
    dev = df[
        df["role"].isin(DEVELOPMENT_ROLES)
    ].copy()

    if dev.empty:
        raise ValueError("No development rows found")

    if dev["tube_id"].isna().any():
        raise ValueError("Development metadata has missing tube IDs")

    if dev["instance_id"].isna().any():
        raise ValueError("Development metadata has missing instance IDs")

    dev["conc"] = pd.to_numeric(
        dev["conc"], errors="raise"
    ).astype(float)

    dev["instance_id"] = pd.to_numeric(
        dev["instance_id"], errors="raise"
    ).astype(int)

    dev["condition"] = (
        dev["device"].astype(str)
        + "_"
        + dev["lamp"].astype(str)
    )

    if not set(dev["instance_id"].unique()).issubset(set(INSTANCES)):
        raise ValueError("Unexpected instance_id found")

    return df, dev


def image_stats(row):
    """Calculate unmasked whole-frame Lab means and alpha fraction."""

    path = image_path(row["image_path"])

    with Image.open(path) as im:
        im.load()

        has_alpha = (
            "A" in im.getbands()
            or "transparency" in im.info
        )

        rgba = np.asarray(
            im.convert("RGBA"),
            dtype=np.uint8,
        )

    alpha = rgba[:, :, 3]

    # Exclude fully transparent pixels from Lab calculation.
    visible = alpha > 0

    if not visible.any():
        raise ValueError(f"No visible pixels in {path}")

    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    lab = rgb2lab(rgb)

    values = lab[visible]

    # Fraction of pixels that are not fully opaque.
    # For ordinary BMP/JPG images this will be zero.
    transparent_fraction = float((alpha < 255).mean())

    return {
        "width": int(rgba.shape[1]),
        "height": int(rgba.shape[0]),
        "mean_L_star": float(values[:, 0].mean()),
        "mean_a_star": float(values[:, 1].mean()),
        "mean_b_star": float(values[:, 2].mean()),
        "transparent_pixel_fraction": transparent_fraction,
        "alpha_channel_present": bool(has_alpha),
    }


def extract_eda(dev):
    rows = []

    for i, (_, row) in enumerate(dev.iterrows(), start=1):

        record = row.to_dict()
        record.update(image_stats(row))
        rows.append(record)

        if i % 100 == 0 or i == len(dev):
            print(
                f"Processed {i}/{len(dev)} development images"
            )

    x = pd.DataFrame(rows)

    x.to_csv(
        OUT / "phase4_image_eda.csv",
        index=False,
    )

    return x


def make_tube_table(x):
    """Aggregate image measurements to one row per tube/condition.

    Each tube has one concentration within one instance.
    Multiple photos from that tube are averaged, so a tube with
    more photographs does not receive more weight in tube-level plots.
    """

    group_cols = [
        "tube_id",
        "conc",
        "instance_id",
        "device",
        "lamp",
        "condition",
    ]

    tube = (
        x.groupby(group_cols, as_index=False)
        .agg(
            tube_mean_L_star=("mean_L_star", "mean"),
            tube_mean_a_star=("mean_a_star", "mean"),
            tube_mean_b_star=("mean_b_star", "mean"),
            n_images=("image_id", "nunique"),
        )
    )

    return tube


# ---------------------------------------------------------
# 4.1 Images per concentration and tubes per concentration
# ---------------------------------------------------------

def plot_counts(x):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        sharex=True,
    )

    for ax, (device, lamp, label) in zip(
        axes.flat, CONDITIONS
    ):

        q = x[
            (x.device == device)
            & (x.lamp == lamp)
        ]

        for instance in INSTANCES:

            z = q[q.instance_id == instance]

            counts = (
                z.groupby("conc")["image_id"]
                .nunique()
                .sort_index()
            )

            if not counts.empty:
                ax.plot(
                    counts.index,
                    counts.values,
                    marker="o",
                    markersize=3,
                    color=INSTANCE_COLORS[instance],
                    label=f"Instance {instance}",
                )

        ax.set(
            title=label,
            ylabel="Development image count",
        )

        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("Concentration (% w/v)")

    fig.suptitle(
        "Images per concentration and condition, grouped by instance\n"
        "Development data only"
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "01_images_per_concentration_condition.png",
        dpi=180,
    )

    plt.close(fig)

    # One row per physical tube.
    tubes = x[
        ["tube_id", "conc", "instance_id"]
    ].drop_duplicates()

    table = (
        tubes.groupby(["conc", "instance_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=INSTANCES, fill_value=0)
        .sort_index()
    )

    ax = table.plot(
        kind="bar",
        stacked=True,
        color=[INSTANCE_COLORS[i] for i in INSTANCES],
        figsize=(14, 6),
    )

    ax.set(
        title="Development tubes per concentration, grouped by instance",
        xlabel="Concentration (% w/v)",
        ylabel="Physical tube count",
    )

    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()

    plt.savefig(
        FIG / "02_tubes_per_concentration.png",
        dpi=180,
    )

    plt.close()


# ---------------------------------------------------------
# 4.2 Contact sheets: five concentrations x four conditions
# ---------------------------------------------------------

def contact_sheet(x, concentration, device, lamp):

    q = x[
        np.isclose(x.conc, concentration)
        & (x.device == device)
        & (x.lamp == lamp)
    ].copy()

    q = q.sort_values(
        ["instance_id", "cohort", "image_id"]
    )

    thumb_width = 300
    thumb_height = 205
    cell_height = 245
    ncol = 3

    nrow = max(1, math.ceil(len(q) / ncol))

    sheet = Image.new(
        "RGB",
        (ncol * thumb_width, 75 + nrow * cell_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    label = (
        f"{concentration:.2f}% | {device} | "
        f"lamp {lamp} | development only"
    )

    draw.text(
        (12, 12),
        label,
        fill="black",
        font=font,
    )

    draw.text(
        (12, 35),
        f"Images: {len(q)}",
        fill="black",
        font=font,
    )

    if q.empty:
        draw.text(
            (15, 100),
            "No development images for this combination",
            fill="black",
            font=font,
        )

    for i, (_, row) in enumerate(q.iterrows()):

        path = image_path(row["image_path"])

        with Image.open(path) as im:
            thumb = ImageOps.contain(
                im.convert("RGB"),
                (thumb_width - 12, thumb_height - 8),
            )

        col = i % ncol
        r = i // ncol

        x0 = col * thumb_width
        y0 = 75 + r * cell_height

        sheet.paste(
            thumb,
            (
                x0 + (thumb_width - thumb.width) // 2,
                y0 + (thumb_height - thumb.height) // 2,
            ),
        )

        caption = (
            f"{row.image_id} | I{row.instance_id} | {row.cohort}"
        )

        draw.text(
            (x0 + 6, y0 + thumb_height + 3),
            caption[:58],
            fill="black",
            font=font,
        )

    sheet.save(
        FIG / (
            f"contact_{concentration:.2f}_"
            f"{device}_{lamp}.jpg"
        ),
        quality=88,
    )


def make_all_contact_sheets(x):
    for concentration in CONTACT_LEVELS:
        for device, lamp, _ in CONDITIONS:
            contact_sheet(
                x, concentration, device, lamp
            )


# ---------------------------------------------------------
# 4.3 Mean L*, a*, b* against concentration
#      One panel per condition, coloured by instance
# ---------------------------------------------------------

def plot_lab_by_instance(tube):

    channels = [
        ("tube_mean_L_star", "Mean L*"),
        ("tube_mean_a_star", "Mean a*"),
        ("tube_mean_b_star", "Mean b*"),
    ]

    fig, axes = plt.subplots(
        4, 3,
        figsize=(17, 15),
        sharex=True,
    )

    for ri, (device, lamp, label) in enumerate(CONDITIONS):

        q = tube[
            (tube.device == device)
            & (tube.lamp == lamp)
        ]

        for ci, (column, ylabel) in enumerate(channels):

            ax = axes[ri, ci]

            for instance in INSTANCES:

                z = q[q.instance_id == instance]

                if z.empty:
                    continue

                ax.scatter(
                    z.conc,
                    z[column],
                    color=INSTANCE_COLORS[instance],
                    s=35,
                    alpha=0.8,
                    label=f"Instance {instance}",
                )

            ax.set(
                title=f"{label}: {ylabel}",
                ylabel=ylabel,
            )

            ax.grid(alpha=0.25)

            if ri == 3:
                ax.set_xlabel("Concentration (% w/v)")

            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Tube-level mean L*, a*, b* against concentration\n"
        "Points coloured by preparation instance; development data only"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(
        FIG / "03_mean_Lab_vs_concentration_by_instance.png",
        dpi=180,
    )

    plt.close(fig)


# ---------------------------------------------------------
# 4.4 Replace timestamp plot:
#     b* against concentration within each instance
# ---------------------------------------------------------

def plot_bstar_by_instance(tube):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        sharex=True,
    )

    for ax, (device, lamp, label) in zip(
        axes.flat, CONDITIONS
    ):

        q = tube[
            (tube.device == device)
            & (tube.lamp == lamp)
        ]

        for instance in INSTANCES:

            z = (
                q[q.instance_id == instance]
                .sort_values("conc")
            )

            if z.empty:
                continue

            ax.plot(
                z.conc,
                z.tube_mean_b_star,
                marker="o",
                markersize=4,
                color=INSTANCE_COLORS[instance],
                label=f"Instance {instance}",
            )

        ax.set(
            title=label,
            xlabel="Concentration (% w/v)",
            ylabel="Tube-level mean b*",
        )

        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Tube-level b* response by preparation instance\n"
        "Timestamp is retained as provenance but not used for grouping"
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "04_bstar_vs_concentration_by_instance.png",
        dpi=180,
    )

    plt.close(fig)


# ---------------------------------------------------------
# 4.5 Transparent-pixel fraction per image, by condition
# ---------------------------------------------------------

def plot_alpha(x):

    order = [
        f"{device}_{lamp}"
        for device, lamp, _ in CONDITIONS
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    rng = np.random.default_rng(42)

    for i, condition in enumerate(order, start=1):

        vals = x.loc[
            x.condition == condition,
            "transparent_pixel_fraction",
        ].dropna().to_numpy()

        if len(vals):
            ax.scatter(
                i + rng.uniform(-0.12, 0.12, len(vals)),
                vals,
                alpha=0.4,
                s=14,
            )

    ax.set_xticks(
        range(1, len(order) + 1),
        order,
    )

    ax.set(
        ylabel="Transparent-pixel fraction per image",
        xlabel="Condition",
        title="Transparent pixels in original images by condition",
    )

    ax.grid(axis="y", alpha=0.25)

    if not x["alpha_channel_present"].any():
        ax.text(
            0.5,
            0.88,
            "No source image has alpha: zero means opaque originals, not a droplet mask.",
            ha="center",
            va="top",
            transform=ax.transAxes,
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
            },
        )

    fig.tight_layout()

    fig.savefig(
        FIG / "05_transparent_pixel_fraction_by_condition.png",
        dpi=180,
    )

    plt.close(fig)


# ---------------------------------------------------------
# L* distribution and bimodality diagnostic
# ---------------------------------------------------------

def plot_Lstar_distribution(x):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
    )

    for ax, (device, lamp, label) in zip(
        axes.flat, CONDITIONS
    ):

        q = x[
            (x.device == device)
            & (x.lamp == lamp)
        ]

        for instance in INSTANCES:

            vals = q.loc[
                q.instance_id == instance,
                "mean_L_star",
            ].dropna()

            if len(vals):
                ax.hist(
                    vals,
                    bins=25,
                    alpha=0.4,
                    density=True,
                    color=INSTANCE_COLORS[instance],
                    label=f"Instance {instance}",
                )

        ax.set(
            title=label,
            xlabel="Per-image whole-frame mean L*",
            ylabel="Density",
        )

        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Whole-frame L* distributions by condition and instance\n"
        "Pre-mask diagnostic; not droplet-only L*"
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "06_Lstar_distribution_by_instance.png",
        dpi=180,
    )

    plt.close(fig)


# ---------------------------------------------------------
# Instance effects at matched concentrations
# ---------------------------------------------------------

def instance_contrasts(tube):

    rows = []

    for device, lamp, _ in CONDITIONS:

        q = tube[
            (tube.device == device)
            & (tube.lamp == lamp)
        ]

        wide = q.pivot_table(
            index="conc",
            columns="instance_id",
            values="tube_mean_b_star",
            aggfunc="mean",
        )

        for inst_a, inst_b in combinations(INSTANCES, 2):

            if inst_a not in wide.columns or inst_b not in wide.columns:
                continue

            common = wide[[inst_a, inst_b]].dropna()

            for concentration, row in common.iterrows():

                rows.append({
                    "device": device,
                    "lamp": lamp,
                    "condition": f"{device}_{lamp}",
                    "instance_a": inst_a,
                    "instance_b": inst_b,
                    "conc": float(concentration),
                    "b_star_instance_a": float(row[inst_a]),
                    "b_star_instance_b": float(row[inst_b]),
                    "difference_b_star_a_minus_b": float(
                        row[inst_a] - row[inst_b]
                    ),
                })

    detailed = pd.DataFrame(rows)

    detailed.to_csv(
        OUT / "phase4_instance_contrasts_by_concentration.csv",
        index=False,
    )

    if detailed.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            detailed.groupby(
                [
                    "condition",
                    "instance_a",
                    "instance_b",
                ],
                as_index=False,
            )
            .agg(
                n_matched_concentrations=("conc", "nunique"),
                mean_difference_b_star=(
                    "difference_b_star_a_minus_b", "mean"
                ),
                sd_difference_b_star=(
                    "difference_b_star_a_minus_b", "std"
                ),
            )
        )

    summary.to_csv(
        OUT / "phase4_instance_contrast_summary.csv",
        index=False,
    )

    return detailed, summary


def plot_instance_contrasts(detailed):

    if detailed.empty:
        print("No matched concentrations for instance comparisons")
        return

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        sharex=True,
    )

    for ax, (device, lamp, label) in zip(
        axes.flat, CONDITIONS
    ):

        q = detailed[
            (detailed.device == device)
            & (detailed.lamp == lamp)
        ]

        pairs = sorted(
            set(zip(q.instance_a, q.instance_b))
        )

        for a, b in pairs:

            z = q[
                (q.instance_a == a)
                & (q.instance_b == b)
            ].sort_values("conc")

            ax.plot(
                z.conc,
                z.difference_b_star_a_minus_b,
                marker="o",
                markersize=4,
                label=f"I{a} - I{b}",
            )

        ax.axhline(0, linewidth=1, color="black")

        ax.set(
            title=label,
            xlabel="Concentration (% w/v)",
            ylabel="Difference in tube-level mean b*",
        )

        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Matched-concentration instance differences in b*\n"
        "Differences are between separate tubes at the same target concentration"
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "09_instance_differences_at_matched_concentrations.png",
        dpi=180,
    )

    plt.close(fig)


# ---------------------------------------------------------
# Saturation and concentration response
# ---------------------------------------------------------

def plot_bstar_response(tube):

    response = (
        tube.groupby(
            ["condition", "device", "lamp", "instance_id", "conc"],
            as_index=False,
        )
        .agg(
            mean_b_star=("tube_mean_b_star", "mean"),
            median_b_star=("tube_mean_b_star", "median"),
            tube_count=("tube_id", "nunique"),
        )
    )

    response.to_csv(
        OUT / "phase4_bstar_response_by_concentration.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        sharex=True,
    )

    for ax, (device, lamp, label) in zip(
        axes.flat, CONDITIONS
    ):

        q = response[
            (response.device == device)
            & (response.lamp == lamp)
        ]

        for instance in INSTANCES:

            z = (
                q[q.instance_id == instance]
                .sort_values("conc")
            )

            if z.empty:
                continue

            ax.plot(
                z.conc,
                z.mean_b_star,
                marker="o",
                markersize=4,
                color=INSTANCE_COLORS[instance],
                label=f"Instance {instance}",
            )

        ax.set(
            title=label,
            xlabel="Concentration (% w/v)",
            ylabel="Mean tube-level b*",
        )

        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "b* response and possible saturation by condition and instance"
    )

    fig.tight_layout()

    fig.savefig(
        FIG / "07_bstar_response_and_saturation.png",
        dpi=180,
    )

    plt.close(fig)

    return response


# ---------------------------------------------------------
# Within-concentration spread vs full response range
# ---------------------------------------------------------

def spread_analysis(tube):

    rows = []

    for condition, q in tube.groupby("condition"):

        by_conc = (
            q.groupby("conc")["tube_mean_b_star"]
            .agg(["mean", "std", "count"])
        )

        repeated = by_conc[
            by_conc["count"] >= 2
        ]

        full_span = float(
            by_conc["mean"].max() - by_conc["mean"].min()
        )

        mean_within_level_sd = (
            float(repeated["std"].mean())
            if len(repeated) else np.nan
        )

        ratio = (
            mean_within_level_sd / full_span
            if full_span > 0 and np.isfinite(mean_within_level_sd)
            else np.nan
        )

        rows.append({
            "condition": condition,
            "n_concentrations": len(by_conc),
            "n_concentrations_with_2plus_tubes": len(repeated),
            "mean_within_concentration_tube_sd": mean_within_level_sd,
            "full_concentration_response_span": full_span,
            "spread_to_response_span_ratio": ratio,
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT / "phase4_within_vs_across_spread.csv",
        index=False,
    )

    ax = result.set_index(
        "condition"
    )["spread_to_response_span_ratio"].plot(
        kind="bar",
        figsize=(10, 5),
    )

    ax.set(
        ylabel="Mean within-level tube SD / full b* response span",
        title="Within-concentration tube spread relative to full response",
    )

    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        FIG / "08_bstar_within_vs_across_spread.png",
        dpi=180,
    )

    plt.close()

    return result


# ---------------------------------------------------------
# Optional descriptive variance decomposition
# ---------------------------------------------------------

def variance_decomposition(x):

    rows = []

    for condition, q in x.groupby("condition"):

        values = q["mean_b_star"].to_numpy(dtype=float)
        grand_mean = float(values.mean())

        total_ss = float(
            np.sum((values - grand_mean) ** 2)
        )

        # Concentration means and counts.
        concentration_stats = (
            q.groupby("conc")["mean_b_star"]
            .agg(["mean", "count"])
        )

        concentration_means = concentration_stats["mean"]

        ss_concentration = float(
            sum(
                row["count"] * (row["mean"] - grand_mean) ** 2
                for _, row in concentration_stats.iterrows()
            )
        )

        # Tube means nested within concentration.
        tube_stats = (
            q.groupby(["conc", "tube_id"])["mean_b_star"]
            .agg(["mean", "count"])
            .reset_index()
        )

        ss_tube_within_concentration = 0.0

        for _, row in tube_stats.iterrows():

            conc_mean = concentration_means.loc[row["conc"]]

            ss_tube_within_concentration += (
                row["count"] * (row["mean"] - conc_mean) ** 2
            )

        # Photo-to-photo differences within each tube.
        ss_photo_within_tube = 0.0

        tube_mean_map = (
            q.groupby("tube_id")["mean_b_star"].mean()
        )

        for _, row in q.iterrows():

            tube_mean = tube_mean_map.loc[row["tube_id"]]

            ss_photo_within_tube += (
                row["mean_b_star"] - tube_mean
            ) ** 2

        reconstructed_ss = (
            ss_concentration
            + ss_tube_within_concentration
            + ss_photo_within_tube
        )

        if not np.isclose(
            reconstructed_ss,
            total_ss,
            rtol=1e-7,
            atol=1e-7,
        ):
            raise AssertionError(
                f"Variance decomposition did not add up for {condition}"
            )

        def fraction(ss):
            return ss / total_ss if total_ss > 0 else np.nan

        rows.append({
            "condition": condition,
            "n_images": len(q),
            "n_concentrations": q["conc"].nunique(),
            "n_tubes": q["tube_id"].nunique(),
            "total_sum_squares": total_ss,
            "ss_between_concentrations": ss_concentration,
            "fraction_between_concentrations": fraction(ss_concentration),
            "ss_between_tubes_within_concentration": (
                ss_tube_within_concentration
            ),
            "fraction_between_tubes_within_concentration": (
                fraction(ss_tube_within_concentration)
            ),
            "ss_between_photos_within_tube": ss_photo_within_tube,
            "fraction_between_photos_within_tube": (
                fraction(ss_photo_within_tube)
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT / "phase4_variance_components_descriptive.csv",
        index=False,
    )

    return result


# ---------------------------------------------------------
# Device and lamp differences, paired within physical tubes
# ---------------------------------------------------------

def device_lamp_differences(tube):

    rows = []

    # Device difference: phone - dino, separately for each lamp.
    for lamp in ["off", "on"]:

        q = tube[tube.lamp == lamp]

        wide = q.pivot_table(
            index=["tube_id", "conc", "instance_id"],
            columns="device",
            values="tube_mean_b_star",
            aggfunc="mean",
        )

        if {"phone", "dino"}.issubset(wide.columns):

            paired = wide.dropna(subset=["phone", "dino"])

            for idx, row in paired.iterrows():

                tube_id, conc, instance = idx

                rows.append({
                    "comparison": f"phone_minus_dino_lamp_{lamp}",
                    "tube_id": tube_id,
                    "conc": conc,
                    "instance_id": instance,
                    "difference_b_star": row["phone"] - row["dino"],
                })

    # Lamp difference: on - off, separately for each device.
    for device in ["dino", "phone"]:

        q = tube[tube.device == device]

        wide = q.pivot_table(
            index=["tube_id", "conc", "instance_id"],
            columns="lamp",
            values="tube_mean_b_star",
            aggfunc="mean",
        )

        if {"on", "off"}.issubset(wide.columns):

            paired = wide.dropna(subset=["on", "off"])

            for idx, row in paired.iterrows():

                tube_id, conc, instance = idx

                rows.append({
                    "comparison": f"lamp_on_minus_off_{device}",
                    "tube_id": tube_id,
                    "conc": conc,
                    "instance_id": instance,
                    "difference_b_star": row["on"] - row["off"],
                })

    detailed = pd.DataFrame(rows)

    detailed.to_csv(
        OUT / "phase4_device_lamp_paired_differences.csv",
        index=False,
    )

    if detailed.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            detailed.groupby(
                ["comparison", "instance_id"],
                as_index=False,
            )
            .agg(
                n_paired_tubes=("tube_id", "nunique"),
                mean_difference_b_star=("difference_b_star", "mean"),
                sd_difference_b_star=("difference_b_star", "std"),
            )
        )

    summary.to_csv(
        OUT / "phase4_device_lamp_paired_summary.csv",
        index=False,
    )

    return detailed, summary


# ---------------------------------------------------------
# Manual image-quality review and inclusion rules
# ---------------------------------------------------------

def create_review_files(x):

    base_columns = [
        "image_id",
        "image_path",
        "cohort",
        "instance_id",
        "tube_id",
        "conc",
        "device",
        "lamp",
        "role",
    ]

    base = x[base_columns].copy()

    for col in MANUAL_QC_COLUMNS:
        base[col] = ""

    # Preserve any manual annotations already entered by the user.
    if QC.exists():

        old = pd.read_csv(QC)

        if old["image_id"].duplicated().any():
            raise ValueError(
                "Duplicate image_id entries found in image_quality_review.csv"
            )

        old = old.set_index("image_id")

        for col in MANUAL_QC_COLUMNS:
            if col in old.columns:
                values = old[col].to_dict()
                base[col] = (
                    base["image_id"].map(values).fillna("")
                )

    base.to_csv(QC, index=False)

    if not RULES.exists():

        RULES.write_text(
            textwrap.dedent("""\
            # Phase 4 image inclusion rules

            Review only development images (train + validation).
            Do not open test images.

            ## Include an image when

            - The original file opens and decodes.
            - The intended droplet is visible.
            - The droplet boundary can be identified.
            - Focus is adequate for colour analysis.
            - The droplet is not severely cropped.
            - Severe glare or obstruction does not hide most of the droplet.
            - The image-to-metadata mapping is unambiguous.

            ## Exclude an image only when

            - The file is corrupt or unreadable.
            - No intended droplet can be identified.
            - Severe crop or blur prevents meaningful ROI identification.
            - Severe obstruction/glare hides most of the droplet.
            - The image-to-label mapping is known to be incorrect or ambiguous.

            ## Do not exclude based on

            - Concentration or unexpected colour.
            - Whether the image improves a future model score.
            - Instance, device or lamp membership alone.
            - A future model prediction error.

            Record every exclusion by image_id and give a reason
            in image_quality_review.csv.

            Blank reviews mean pending review. They do not mean
            automatically included or excluded.

            Original BMP/JPG photographs are used, not transparent
            cut-outs. Alpha transparency describes the source file
            only, not droplet segmentation quality.

            Droplet ROI occupancy and mask quality will be assessed
            in Phase 5.
            """),
            encoding="utf-8",
        )


# ---------------------------------------------------------
# Observation template
# ---------------------------------------------------------

def create_observation_template(x):

    if OBS.exists():
        print(
            "Existing phase4_observations.md preserved; "
            "not overwriting manual observations."
        )
        return

    instance_counts = (
        x.groupby("instance_id")["image_id"]
        .nunique()
        .to_dict()
    )

    alpha_present = bool(x["alpha_channel_present"].any())

    if alpha_present:
        alpha_note = (
            "Some images contain alpha information. "
            "Interpret the transparent-pixel fraction per image."
        )
    else:
        alpha_note = (
            "No source image has an alpha channel. "
            "Zero transparent fraction means opaque BMP/JPG originals; "
            "it does not measure droplet occupancy."
        )

    OBS.write_text(
        f"""# Phase 4 observations

Development images analysed: {len(x)} (train + validation).
Test images excluded.

Development image counts by instance: {instance_counts}.

Timestamps remain in metadata for provenance but are not used
as the primary grouping variable or in Phase 4 plots.

## Questions and one-line answers

1. **Is L* bimodal?**
   Review `06_Lstar_distribution_by_instance.png`.
   These are unmasked whole-frame measurements.
   Answer: [state whether modes are visible and whether they align with instance/device/lamp].

2. **Are there preparation-instance differences?**
   Review `phase4_instance_contrasts_by_concentration.csv`,
   `phase4_instance_contrast_summary.csv` and
   `09_instance_differences_at_matched_concentrations.png`.
   Answer: [state which instance differences appear at matched concentrations and conditions].

3. **Is there temporal drift?**
   Not assessed in this phase. Timestamp is retained as provenance,
   but instance is used for preparation-batch comparisons.
   Instance differences do not establish temporal drift.
   Answer: Temporal drift within an instance is not assessed.

4. **Does the response saturate?**
   Review `07_bstar_response_and_saturation.png` separately by
   condition and instance.
   Answer: [state whether b* flattens at high concentrations].

5. **How large is the spread at one concentration compared with
   the change across the whole concentration range?**
   Review `phase4_within_vs_across_spread.csv` and
   `08_bstar_within_vs_across_spread.png`.
   Answer: [report the ratios and their interpretation].

6. **Are any images unusable?**
   Complete `image_quality_review.csv` using `inclusion_rules.md`.
   Answer: [number of excluded images, image IDs and reasons].

7. **(Should) Variance decomposition.**
   Review `phase4_variance_components_descriptive.csv`,
   `phase4_device_lamp_paired_differences.csv` and
   `phase4_device_lamp_paired_summary.csv`.
   Answer: [describe variation between concentrations, tubes/instances,
   photos within tubes, and paired device/lamp differences].

## Alpha-channel result

{alpha_note}

## Interpretation limitations

Each instance has one physical tube per concentration. Therefore,
between-tube variation at a fixed concentration is also variation
between preparation instances. These effects cannot be independently
identified at a fixed concentration.

The November 4 cohort is included or excluded from Phase 4 strictly
according to the current tube-level train/validation/test role.
The test role is excluded even though the cohort name contains
the word "validation".

All Lab measurements in Phase 4 are whole-frame measurements
including background. They are diagnostic only. Whether L* modes
collapse after droplet masking can only be assessed after Phase 5.
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    for path in [FIG, OUT, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)

    metadata, dev = read_metadata()

    test_count = int((metadata["role"] == "test").sum())

    print("PHASE 4 — INSTANCE-BASED EXPLORATION")
    print("=" * 60)
    print(f"Total metadata rows: {len(metadata)}")
    print(f"Development images: {len(dev)}")
    print(f"Locked test images excluded: {test_count}")

    print("\nDevelopment images by instance:")
    print(
        dev.groupby("instance_id")["image_id"]
        .nunique()
        .to_string()
    )

    print("\nDevelopment tubes by instance:")
    print(
        dev[["tube_id", "instance_id"]]
        .drop_duplicates()
        .groupby("instance_id")
        .size()
        .to_string()
    )

    x = extract_eda(dev)
    tube = make_tube_table(x)

    # Required Phase 4 figures.
    plot_counts(x)
    make_all_contact_sheets(x)
    plot_lab_by_instance(tube)
    plot_bstar_by_instance(tube)
    plot_alpha(x)

    # Additional scientific diagnostics.
    plot_Lstar_distribution(x)

    detailed_instance, summary_instance = instance_contrasts(tube)
    plot_instance_contrasts(detailed_instance)

    plot_bstar_response(tube)
    spread_analysis(tube)

    variance_decomposition(x)
    device_lamp_differences(tube)

    # Quality review, inclusion criteria and observation template.
    create_review_files(x)
    create_observation_template(x)

    print("\nPHASE 4 OUTPUTS GENERATED")
    print("-" * 60)
    print("Figures and contact sheets:", FIG.relative_to(ROOT))
    print("Image-level measurements:", (OUT / "phase4_image_eda.csv").relative_to(ROOT))
    print("Instance contrasts:", (OUT / "phase4_instance_contrasts_by_concentration.csv").relative_to(ROOT))
    print("Spread summary:", (OUT / "phase4_within_vs_across_spread.csv").relative_to(ROOT))
    print("Variance decomposition:", (OUT / "phase4_variance_components_descriptive.csv").relative_to(ROOT))
    print("Image quality review:", QC.relative_to(ROOT))
    print("Inclusion rules:", RULES.relative_to(ROOT))
    print("Observation template:", OBS.relative_to(ROOT))

    print("\nNo test images were opened.")
    print("Timestamps were not used for Phase 4 grouping.")
    print(
        "Review the figures and complete phase4_observations.md "
        "before declaring Phase 4 complete."
    )


if __name__ == "__main__":
    main()
