import copy

import jax.numpy as jnp
from matplotlib.testing.compare import compare_images

from cabinetry.contrib import matplotlib_visualize


def test_data_MC(tmp_path):
    fname = tmp_path / "fig.pdf"
    histo_dict_list = [
        {
            "label": "Background",
            "isData": False,
            "yields": jnp.asarray([12.5, 14]),
            "variable": "x",
        },
        {
            "label": "Signal",
            "isData": False,
            "yields": jnp.asarray([2, 5]),
            "variable": "x",
        },
        {
            "label": "Data",
            "isData": True,
            "yields": jnp.asarray([13, 15]),
            "variable": "x",
        },
    ]
    total_model_unc = jnp.sqrt([0.17, 0.29])
    bin_edges = jnp.asarray([1, 2, 3])
    matplotlib_visualize.data_MC(histo_dict_list, total_model_unc, bin_edges, fname)
    assert compare_images("tests/contrib/reference/data_MC.pdf", str(fname), 0) is None
    fname.unlink()  # delete figure

    histo_dict_list_log = copy.deepcopy(histo_dict_list)
    histo_dict_list_log[0]["yields"] = jnp.asarray([2000, 14])
    histo_dict_list_log[2]["yields"] = jnp.asarray([2010, 15])
    total_model_unc_log = jnp.asarray([50, 1.5])
    bin_edges_log = jnp.asarray([10, 100, 1000])
    fname_log = fname.with_name(fname.stem + "_log" + fname.suffix)

    # automatic log scale
    matplotlib_visualize.data_MC(
        histo_dict_list_log, total_model_unc_log, bin_edges_log, fname, log_scale_x=True
    )
    assert (
        compare_images("tests/contrib/reference/data_MC_log.pdf", str(fname_log), 0)
        is None
    )
    fname_log.unlink()

    # linear scale forced
    matplotlib_visualize.data_MC(
        histo_dict_list, total_model_unc, bin_edges, fname, log_scale=False
    )
    assert compare_images("tests/contrib/reference/data_MC.pdf", str(fname), 0) is None
    fname.unlink()

    # log scale forced
    matplotlib_visualize.data_MC(
        histo_dict_list_log,
        total_model_unc_log,
        bin_edges_log,
        fname,
        log_scale=True,
        log_scale_x=True,
    )
    assert (
        compare_images("tests/contrib/reference/data_MC_log.pdf", str(fname_log), 0)
        is None
    )
    fname_log.unlink()


def test_correlation_matrix(tmp_path):
    fname = tmp_path / "fig.pdf"
    # one parameter is below threshold so no text is shown for it on the plot
    corr_mat = jnp.asarray([[1.0, 0.35, 0.002], [0.35, 1.0, -0.2], [0.002, -0.2, 1.0]])
    labels = ["a", "b", "c"]
    matplotlib_visualize.correlation_matrix(corr_mat, labels, fname)
    assert (
        compare_images("tests/contrib/reference/correlation_matrix.pdf", str(fname), 0)
        is None
    )


def test_pulls(tmp_path):
    fname = tmp_path / "fig.pdf"
    bestfit = jnp.asarray([-0.2, 0.0, 0.1])
    uncertainty = jnp.asarray([0.9, 1.0, 0.7])
    labels = jnp.asarray(["a", "b", "c"])
    matplotlib_visualize.pulls(bestfit, uncertainty, labels, fname)
    assert compare_images("tests/contrib/reference/pulls.pdf", str(fname), 0) is None


def test_ranking(tmp_path):
    fname = tmp_path / "fig.pdf"
    bestfit = jnp.asarray([0.3, -0.1])
    uncertainty = jnp.asarray([0.8, 1.0])
    labels = jnp.asarray(["jet energy scale", "modeling uncertainty"])
    impact_prefit_up = jnp.asarray([0.5, 0.3])
    impact_prefit_down = jnp.asarray([-0.4, -0.3])
    impact_postfit_up = jnp.asarray([0.4, 0.25])
    impact_postfit_down = jnp.asarray([-0.3, -0.25])

    matplotlib_visualize.ranking(
        bestfit,
        uncertainty,
        labels,
        impact_prefit_up,
        impact_prefit_down,
        impact_postfit_up,
        impact_postfit_down,
        fname,
    )
    assert compare_images("tests/contrib/reference/ranking.pdf", str(fname), 0) is None


def test_templates(tmp_path):
    fname = tmp_path / "fig.pdf"
    nominal_histo = {
        "yields": jnp.asarray([1.0, 1.2]),
        "stdev": jnp.asarray([0.05, 0.06]),
    }
    up_histo_orig = {
        "yields": jnp.asarray([1.2, 1.7]),
        "stdev": jnp.asarray([0.05, 0.07]),
    }
    up_histo_mod = {
        "yields": jnp.asarray([1.3, 1.6]),
        "stdev": jnp.asarray([0.05, 0.07]),
    }
    down_histo_orig = {
        "yields": jnp.asarray([0.9, 0.9]),
        "stdev": jnp.asarray([0.06, 0.07]),
    }
    down_histo_mod = {
        "yields": jnp.asarray([0.85, 0.95]),
        "stdev": jnp.asarray([0.06, 0.07]),
    }
    bin_edges = jnp.asarray([0.0, 1.0, 2.0])
    variable = "x"

    matplotlib_visualize.templates(
        nominal_histo,
        up_histo_orig,
        up_histo_mod,
        down_histo_orig,
        down_histo_mod,
        bin_edges,
        variable,
        fname,
    )
    assert (
        compare_images("tests/contrib/reference/templates.pdf", str(fname), 0) is None
    )

    # only single variation specified
    matplotlib_visualize.templates(
        nominal_histo,
        up_histo_orig,
        up_histo_mod,
        {},
        {},
        bin_edges,
        variable,
        fname,
    )


def test_scan(tmp_path):
    fname = tmp_path / "fig.pdf"
    par_name = "a"
    par_mle = 1.5
    par_unc = 0.2
    par_vals = jnp.asarray([1.1, 1.3, 1.5, 1.7, 1.9])
    par_nlls = jnp.asarray([4.1, 1.0, 0.0, 1.1, 3.9])

    matplotlib_visualize.scan(par_name, par_mle, par_unc, par_vals, par_nlls, fname)
    assert compare_images("tests/contrib/reference/scan.pdf", str(fname), 0) is None

    # no 68% CL / 95% CL text
    par_nlls = jnp.asarray([0.1, 0.04, 0.0, 0.04, 0.1])
    matplotlib_visualize.scan(par_name, par_mle, par_unc, par_vals, par_nlls, fname)


def test_limit(tmp_path):
    fname = tmp_path / "fig.pdf"
    observed_CLs = jnp.asarray([0.31, 0.05, 0.005, 0.0001])
    expected_CLs = jnp.asarray(
        [
            [0.07, 0.16, 0.35, 0.62, 0.88],
            [0.0, 0.01, 0.06, 0.23, 0.56],
            [0.0, 0.0, 0.0, 0.05, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.04],
        ]
    )
    poi_values = jnp.asarray([0.5, 1.0, 1.5, 2.0])

    matplotlib_visualize.limit(observed_CLs, expected_CLs, poi_values, fname)
    assert compare_images("tests/contrib/reference/limit.pdf", str(fname), 0) is None
