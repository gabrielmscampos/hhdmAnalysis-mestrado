import anatools.data as data


def signal_label(param_0, param_1):
    label = r"$m_H=$" + str(param_0) + r", $m_\mathit{a}=$" + str(param_1)
    return label


def stack_sorting(df, colors_list, labels_list, bkg_list):
    dataframes = [df[key] for key in bkg_list]
    dataframes, labels, colors, sizes = data.order_datasets(
        dataframes, labels_list, colors_list
    )
    return dataframes, labels, colors


def position(gs1, grid, main, sub):
    """
    Auxiliar function to compute the position of the subplot
    Args:
        gs1 (matplotlib.pyplot.GridSpec): GridSpec object from matplotlib
        grid (list): List representation of subplot grid [rows, cols]
        main (int): Subplot column number
        sub (int): Subplot row number
    Returns:
        [type]: [description]
    """
    return gs1[(main - 1) + (sub - 1) * grid[1]]


def process_signals(sigs):
    return [
        {
            "key": sig[0],
            "label": signal_label(sig[0].split("_")[1], sig[0].split("_")[2]),
            "color": sig[1],
        }
        for sig in sigs
    ]
