## graphs page
import numpy as np
import pandas as pd
import plotly.express as px
import os
from helpers import *
from datetime import datetime


def wide_to_long(df, id_var="Date", columns=[]):
    dfs = {column: df[[id_var, column]] for column in columns}

    for column, df_column in dfs.items():
        df_column["line"] = column
        df_column["data"] = df_column[column]
        df_column.drop(column, axis=1)
    return pd.concat([df[["Date", "line", "data"]] for _, df in dfs.items()], ignore_index=True)


def plot_line(df, title="", freq=10, y=["Close", "Close_benchmark"]):
    fig = px.line(df[::freq], x="Date", y=y,
                  title=title)
    return fig


def log_returns(df):
    """Calculates rolling compounded returns"""
    df["Close"] = np.log(df["Close"] + 1)
    df["Close_benchmark"] = np.log(df["Close_benchmark"] + 1)
    return df


def compsum_array(returns):
    """Calculates rolling compounded returns"""
    return returns.add(1).cumprod() - 1


def sum_array(returns):
    """Calculates rolling compounded returns"""
    return returns.cumsum()


def compsum_df(df):
    """Calculates rolling compounded returns"""
    df["Close"] = compsum_array(df["Close"])
    df["Close_benchmark"] = compsum_array(df["Close_benchmark"])
    return df


def match_volatility(df):
    bmark_vol = df["Close_benchmark"].std()
    df["Close"] = (df["Close"] / df["Close"].std()) * bmark_vol
    return df


def create_returns_graph(filepath, start_date=None, end_date=None):
    df = pd.read_csv(os.path.join(filepath, "returns.csv"))
    df.fillna(0, inplace=True)
    df = filters_date(df, start_date=start_date, end_date=end_date)
    fig = plot_line(df)
    return fig


def make_cumulative_returns_graph(filepath, start_date=None, end_date=None):
    df = pd.read_csv(os.path.join(filepath, "returns.csv"))
    df.fillna(0, inplace=True)
    df = compsum_df(filters_date(df, start_date=start_date, end_date=end_date))
    fig = plot_line(df, title="Cumulative return")
    # fig.show()
    return fig
    # df = filters_date(df, start_date=start_date, end_date=end_date)


def make_log_cumulative_returns_graph(filepath, start_date=None, end_date=None):
    df = pd.read_csv(os.path.join(filepath, "returns.csv"))
    df.fillna(0, inplace=True)
    df = log_returns(df)
    df = compsum_df(filters_date(df, start_date=start_date, end_date=end_date))
    fig = plot_line(df, title="Log cumulative return")
    # fig.show()
    return fig


def cumulative_returns_volatility_graph(filepath, start_date=None, end_date=None):
    df = pd.read_csv(os.path.join(filepath, "returns.csv"))
    df.fillna(0, inplace=True)
    df = match_volatility(df)
    df = compsum_df(filters_date(df, start_date=start_date, end_date=end_date))
    fig = plot_line(df, title="Cumulative return (Volatility matched)")
    # fig.show()
    return fig


def rolling_sharpe(filepath, start_date=None, end_date=None):
    df = pd.read_csv(os.path.join(filepath, "rolling_sharpe.csv"))
    df.fillna(0, inplace=True)
    df = filters_date(df, start_date=start_date, end_date=end_date)
    df["Mean"] = df['Close'].mean()
    fig = plot_line(df,
                    title="Rolling Sharpe (6 months)",
                    y=['Close', "Mean"])
    # fig.show()
    return fig


def rolling_sortino(filepath, start_date=None, end_date=None):
    df = pd.read_csv(os.path.join(filepath, "rolling_sortino.csv"))
    df.fillna(0, inplace=True)
    df = filters_date(df, start_date=start_date, end_date=end_date)
    df["Mean"] = df['Close'].mean()
    fig = plot_line(df,
                    title="Rolling Sortino (6 months)",
                    y=['Close', "Mean"])
    # fig.show()
    return fig


import quantstats.stats as _stats
import quantstats.plots as _plots
import matplotlib.pyplot as _plt
import seaborn as _sns


def monthly_heatmap(returns, annot_size=10, figsize=(10, 5),
                    cbar=True, square=False,
                    compounded=True, eoy=False,
                    grayscale=False, fontname='Arial',
                    ylabel=True, savefig=None, show=True, filepath=None):
    # colors, ls, alpha = _core._get_colors(grayscale)
    cmap = 'gray' if grayscale else 'RdYlGn'

    returns = _stats.monthly_returns(returns, eoy=eoy,
                                     compounded=compounded) * 100

    fig_height = len(returns) / 3

    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0] * 1.04, max([fig_height, figsize[1]]))

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title('      Monthly Returns (%)\n', fontsize=14, y=.995,
                 fontname=fontname, fontweight='bold', color='black')
    return returns
    #print(returns)
    # _sns.set(font_scale=.9)
    ax = _sns.heatmap(returns, ax=ax, annot=True, center=0,
                      annot_kws={"size": annot_size},
                      fmt="0.2f", linewidths=0.5,
                      square=square, cbar=cbar, cmap=cmap,
                      cbar_kws={'format': '%.0f%%'})
    # _sns.set(font_scale=1)

    # align plot to match other
    if ylabel:
        ax.set_ylabel('Years', fontname=fontname,
                      fontweight='bold', fontsize=12)
        ax.yaxis.set_label_coords(-.1, .5)

    ax.tick_params(colors="#808080")
    _plt.xticks(rotation=0, fontsize=annot_size * 1.2)
    _plt.yticks(rotation=0, fontsize=annot_size * 1.2)

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def make_heatmap(filepath, start_date=None, end_date=None):
    returns = pd.read_csv(os.path.join(filepath, "returns.csv"))
    returns.index = [string_to_time(string, "%Y-%m-%d") for string in returns.Date]
    data = monthly_heatmap(returns)
    fig = px.imshow(data, aspect="auto", text_auto=".2f", title="Monthly Heatmap",
                    color_continuous_scale="RdBu")
    # fig.show()
    fig.update_xaxes(side="bottom", tickangle=0)
    return fig