import quantstats_modified as qs_mod
import os
qs_mod.extend_pandas()

def make_files(stock, etf, filepath, start_date=None, end_date=None, st=None):
    #stock = qs_mod.utils.download_returns(stock)
    #qs_mod.stats.sharpe(stock)
    #stock.sharpe()
    #qs_mod.reports.html(stock,
    #                etf,
    #                start_date,
    #                end_date,
    #                output=os.path.join(os.getcwd(),'quantstats-tearsheet3.html'))
    st.text(stock)
    st.text(etf)
    st.text(start_date)
    st.text(end_date)

    stock = qs_mod.utils.download_returns(stock)
    qs_mod.stats.sharpe(stock)
    stock.sharpe()

    # before this runs - it checks if this stock data is saved already
    # if not saved - run and make the data
    qs_mod.old_reports.html(stock, etf,
                            start_date=None,
                            end_date=None,
                            output=os.path.join(filepath,
                                        'quantstats-tearsheet3.html'),
                            filepath=filepath)
    # if it is saved - do nothing

