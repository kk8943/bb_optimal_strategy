import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import quantrix.modules as qx


class strategy_():
    def __init__(self, sigma, x0, n_paths, xt=0, T=33, granularity=1):
        self.sigma = sigma
        self.x0 = x0
        self.xt = xt
        self.T = T
        self.n_paths = n_paths
        self.granularity = granularity

    def simulation(self):
        pass

    def time_linspace(self):
        time_ = np.linspace(0, self.T, self.T * self.granularity + 1)
        return time_

    def expectation(self):
        pass

    def vol(self):
        pass

    def naive_str(self):
        return np.array([self.x0 / self.T] * self.T)


class standart_bb(strategy_):
    def __init__(self, sigma, x0, n_paths, T=33):
        super().__init__(sigma, x0, n_paths, T)

    def simulation(self):
        w = np.array(qx.bm_simulations(self.n_paths, self.granularity, self.T))
        bridge = [self.sigma * (w[i] - self.time_linspace() / self.T * w[i][-1]) for i in range(self.n_paths)]
        X = self.x0 * (1 - self.time_linspace() / self.T) + bridge
        return X

    def expectation(self):
        means_ = [np.mean(i) for i in self.simulation().T]
        return means_

    def vol(self):
        vola = [np.std(i) for i in self.simulation().T]
        return vola

    def simulation_adjusted(self):
        pass


class bb_with_bounds_simulation(standart_bb):
    def __init__(self, sigma, x0, n_paths, T=33):
        super().__init__(sigma, x0, n_paths, T)

    def simulation_adjusted(self):
        bb = self.simulation()

        X = []
        for i in range(self.n_paths):
            bi = bb[i]
            for j in range(len(bi) - 1):
                if bi[j] <= 0:
                    bi = list(bi)[:j] + [0] * (len(bi) - j)
                    pass
                elif bi[j] >= self.x0 and j > 0:
                    bi = list(bi)[:j] + [2 - point for point in bi[j:]]
                    pass
            X.append(bi)

        return np.array(X)

    def expectation(self):
        means_ = [np.mean(i) for i in self.simulation_adjusted().T]
        return means_

    def vol(self):
        vola = [np.std(i) for i in self.simulation_adjusted().T]
        return vola


class bb_with_low_bound_strategy(standart_bb):
    def __init__(self, sigma, x0, n_paths, T=33):
        super().__init__(sigma, x0, n_paths, T)

    def simulation_adjusted(self):
        bb = self.simulation()

        X = []
        for i in range(self.n_paths):
            bi = bb[i]
            for j in range(len(bi) - 1):
                if bi[j] <= 0:
                    bi = list(bi)[:j] + [0] * (len(bi) - j)
                    pass
            X.append(bi)

        return np.array(X)

    def expectation(self):
        means_ = [np.mean(i) for i in self.simulation_adjusted().T]
        return means_

    def vol(self):
        vola = [np.std(i) for i in self.simulation_adjusted().T]
        return vola


class prices():
    def __init__(self, prices, h=0.08):
        self.S = prices
        self.h = h
        self.x_ = np.linspace(0, 1, len(self.S))

    def _gaus_kernel(self, x, xi):
        return np.exp(-0.5 * (x - xi) ** 2 / self.h ** 2)

    def _kd_regression(self, x):
        k_h = self._gaus_kernel(x, self.x_)
        numerator = np.dot(self.S, k_h)
        denominator = np.sum(k_h)
        return numerator / denominator

    def kde(self):
        y_estimated = [self._kd_regression(xi) for xi in self.x_]
        return np.array(y_estimated)

    def plot(self):
        gr = go.Figure()
        gr.add_trace(go.Scatter(x=self.x_, y=self.kde(), name='KDE'))
        gr.add_trace(go.Scatter(x=self.x_, y=self.S, name='real'))
        gr.update_layout(
            title_text='Approximation of historical prices behavior during a trading day.',
            title_font=dict(size=18, color='#1a1a1a'),
            title_x=0.05,
            title_xanchor='left',
            title_pad=dict(t=20, b=10),
            width=1200,
            height=1600,
            margin=dict(l=80, r=50, t=90, b=70),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=14, color='#333333'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=14),
            ),
        )
        gr.update_xaxes(title_text='Time partitions', title_font=dict(size=14), showline=True, linecolor='black', linewidth=1, mirror=True)
        gr.update_yaxes(title_text='Price', title_font=dict(size=14), showline=True, linecolor='black', linewidth=1, mirror=True)
        gr.show()


class revenues():
    def __init__(self, predicted_quantity, price):
        self.predicted_quantity = predicted_quantity
        self.price = price

    def sold_(self):
        sold_ = np.array([abs(self.predicted_quantity[i] - self.predicted_quantity[i - 1]) for i in range(1, len(self.predicted_quantity))])
        return sold_


class calibration(strategy_):
    def __init__(self, sigma, x0, n_paths, T=33):
        super().__init__(sigma, x0, n_paths, T)
        self.sets = None

    def _split_data(self, data):
        training_data = data[:int(0.8 * len(data))]
        test_set = data[int(0.8 * len(data)):]
        return training_data, test_set

    def _group_by_days(self, data):
        self.sets = self._split_data(data)
        training_data = self.sets[0].groupby(by='begin_time')['inventory_left']
        test_set = self.sets[1].groupby(by='begin_time')['inventory_left']
        return training_data, test_set

    def working_data_(self, data):
        working_data = self._group_by_days(data)[0]
        working_data = [1] + [np.mean(w[1]) for w in working_data]
        return working_data

    def plot(self, data):
        working_data = self.working_data_(data)
        plot = go.Figure()
        plot.add_trace(go.Scatter(x=self.time_linspace(), y=working_data))
        plot.update_layout(
            title_text='Calibration working data',
            title_font=dict(size=18, color='#1a1a1a'),
            title_x=0.05,
            title_xanchor='left',
            title_pad=dict(t=20, b=10),
            width=1200,
            height=1600,
            margin=dict(l=80, r=50, t=90, b=70),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=14, color='#333333'),
        )
        plot.update_xaxes(title_text='Time partitions', title_font=dict(size=14), showline=True, linecolor='black', linewidth=1, mirror=True)
        plot.update_yaxes(title_text='Inventory left', title_font=dict(size=14), showline=True, linecolor='black', linewidth=1, mirror=True)
        plot.show()

    def calibrate(self, data):
        return np.std(self.working_data_(data))


def moments(el):
    k1 = el
    setka = np.arange(0, 1.1, 0.1)
    moments = {}
    t = float('inf')
    m = float('inf')
    for tau in range(len(k1)):
        for i in setka[::-1]:
            if k1[tau] <= i:
                if k1[tau] <= m:
                    moments[tau] = k1[tau]
                    t = tau
                    m = k1[tau]
            else:
                moments[tau] = m
    return moments


def _build_price_series(historic_data):
    grouped = historic_data.groupby(by='begin_time')['close'].mean()
    return np.array(grouped)


def _revenue_from_inventory_path(inventory_path, kde_prices):
    sold_quantity = np.abs(np.diff(inventory_path))
    valid_size = min(len(sold_quantity), len(kde_prices))
    price_slice = kde_prices[:valid_size]
    sold_quantity = sold_quantity[:valid_size]
    return np.sum(sold_quantity * price_slice)


def _moments_strategy_stats(adjusted_paths, kde_prices):
    revenue_paths = []
    for path in adjusted_paths:
        moment_path = moments(path)
        x_ = list(moment_path.keys())
        y_ = list(moment_path.values())
        sold_quantity = np.abs(np.diff(y_))
        valid_indexes = [xi for xi in x_[1:] if 0 < xi <= len(kde_prices)]
        price_slice = np.array([kde_prices[xi - 1] for xi in valid_indexes])
        qty_slice = sold_quantity[: len(price_slice)]
        revenue_paths.append((np.sum(qty_slice * price_slice), x_, y_))

    best_revenue, best_x, best_y = max(revenue_paths, key=lambda item: item[0])
    mean_inventory = np.mean(np.array([path[2] for path in revenue_paths]), axis=0)
    mean_revenue = _revenue_from_inventory_path(mean_inventory, kde_prices)

    return best_revenue, best_x, best_y, mean_revenue, mean_inventory


def evaluate_strategies(historic_data, x0=1, n_paths=1000):
    sigma = calibration(sigma=0, x0=x0, n_paths=n_paths).calibrate(historic_data)
    s_t = _build_price_series(historic_data)
    kde_prices = prices(s_t).kde()

    bb_st = standart_bb(sigma=sigma, x0=x0, n_paths=n_paths)
    bb_low = bb_with_low_bound_strategy(sigma=sigma, x0=x0, n_paths=n_paths)

    vwap_inventory = np.linspace(x0, 0, bb_st.T * bb_st.granularity + 1)
    vwap_revenue = _revenue_from_inventory_path(vwap_inventory, kde_prices)

    standard_bb_revenue = _revenue_from_inventory_path(bb_st.expectation(), kde_prices)
    low_bound_revenue = _revenue_from_inventory_path(bb_low.expectation(), kde_prices)
    moments_revenue, moment_x, moment_y, moment_mean_revenue, moment_mean_inventory = _moments_strategy_stats(
        bb_low.simulation_adjusted(), kde_prices
    )

    results = pd.DataFrame([
        {'strategy_name': 'VWAP uniformly selling', 'total_revenue': vwap_revenue},
        {'strategy_name': 'Standard BB', 'total_revenue': standard_bb_revenue},
        {'strategy_name': 'BB + low bound + moments (mean)', 'total_revenue': moment_mean_revenue},
        {'strategy_name': 'BB + low bound + moments (best)', 'total_revenue': moments_revenue},
    ])

    # Prepare 10 sample moment-based trajectories
    moment_paths_10 = []
    adjusted_paths = bb_low.simulation_adjusted()
    for path in adjusted_paths[:10]:
        moment_path = moments(path)
        x_values = list(moment_path.keys())
        y_values = list(moment_path.values())
        moment_paths_10.append((x_values, y_values))

    # Calculate non-cumulative sold quantities for each strategy
    vwap_sold = np.abs(np.diff(vwap_inventory))
    standard_bb_sold = np.abs(np.diff(bb_st.expectation()))
    moments_sold = np.abs(np.diff(moment_y))
    moment_mean_sold = np.abs(np.diff(moment_mean_inventory))

    report = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            'All strategies comparison',
            'BB low bound + moments: 10 sample trajectories',
            'Non-cumulative sold quantity paths for each strategy',
        ],
        vertical_spacing=0.12,
    )

    report.add_trace(
        go.Scatter(
            x=np.arange(len(vwap_inventory)),
            y=vwap_inventory,
            name='VWAP uniform selling',
            line=dict(color='black', dash='dash'),
            mode='lines+markers',
        ),
        row=1,
        col=1,
    )
    report.add_trace(
        go.Scatter(
            x=np.arange(len(bb_st.expectation())),
            y=bb_st.expectation(),
            name='Standard BB mean',
            line=dict(color='red'),
            mode='lines+markers',
        ),
        row=1,
        col=1,
    )
    report.add_trace(
        go.Scatter(
            x=np.arange(len(bb_low.expectation())),
            y=bb_low.expectation(),
            name='Low bound BB mean',
            line=dict(color='green'),
            mode='lines+markers',
        ),
        row=1,
        col=1,
    )
    report.add_trace(
        go.Scatter(
            x=moment_x,
            y=moment_mean_inventory,
            name='BB low bound + moments mean',
            line=dict(color='purple'),
            mode='lines+markers',
        ),
        row=1,
        col=1,
    )
    report.add_trace(
        go.Scatter(
            x=moment_x,
            y=moment_y,
            name='BB low bound + moments best revenue',
            line=dict(color='magenta', dash='dot'),
            mode='lines+markers',
        ),
        row=1,
        col=1,
    )

    for idx, (x_values, y_values) in enumerate(moment_paths_10, start=1):
        report.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=f'Path {idx}',
                opacity=0.7,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    report.add_trace(
        go.Scatter(
            x=np.arange(1, len(vwap_inventory)),
            y=vwap_sold,
            name='VWAP sold quantity',
            line=dict(color='black', dash='dash'),
            mode='lines+markers',
        ),
        row=3,
        col=1,
    )
    report.add_trace(
        go.Scatter(
            x=np.arange(1, len(bb_st.expectation())),
            y=standard_bb_sold,
            name='Standard BB sold quantity',
            line=dict(color='red'),
            mode='lines+markers',
        ),
        row=3,
        col=1,
    )
    report.add_trace(
        go.Scatter(
            x=np.array(moment_x[1:]),
            y=moments_sold,
            name='BB low bound + moments sold quantity (best revenue)',
            line=dict(color='magenta', dash='dot'),
            mode='lines+markers',
        ),
        row=3,
        col=1,
    )
    report.add_trace(
        go.Scatter(
            x=np.array(moment_x[1:]),
            y=moment_mean_sold,
            name='BB low bound + moments sold quantity (mean trajectory)',
            line=dict(color='purple', dash='dash'),
            mode='lines+markers',
        ),
        row=3,
        col=1,
    )

    report.update_xaxes(title_text='Time partitions', title_font=dict(size=14), row=1, col=1, showline=True, linecolor='black', linewidth=1, mirror=True)
    report.update_xaxes(title_text='Time partitions', title_font=dict(size=14), row=2, col=1, showline=True, linecolor='black', linewidth=1, mirror=True)
    report.update_xaxes(title_text='Time partitions', title_font=dict(size=14), row=3, col=1, showline=True, linecolor='black', linewidth=1, mirror=True)
    report.update_yaxes(title_text='Inventory left', title_font=dict(size=14), row=1, col=1, showline=True, linecolor='black', linewidth=1, mirror=True)
    report.update_yaxes(title_text='Inventory left', title_font=dict(size=14), row=2, col=1, showline=True, linecolor='black', linewidth=1, mirror=True)
    report.update_yaxes(title_text='Sold quantity', title_font=dict(size=14), row=3, col=1, showline=True, linecolor='black', linewidth=1, mirror=True)
    report.update_layout(
        title_text=f'Strategy evaluation report (σ={sigma:.4f})',
        title_font=dict(size=18, color='#1a1a1a'),
        title_x=0.05,
        title_xanchor='left',
        title_pad=dict(t=20, b=10),
        width=1200,
        height=1600,
        margin=dict(l=80, r=50, t=90, b=70),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14, color='#333333'),
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
    )

    pio.write_html(report, file='strategy_evaluation_report.html', auto_open=False, full_html=True, include_plotlyjs='cdn')
    report.show()

    print('Saved strategy report to strategy_evaluation_report.html')
    return results


if __name__ == '__main__':
    data0 = pd.read_csv('CNYRUB_TOM_10m.csv', sep=',')
    data = data0[['open', 'close', 'volume', 'begin', 'end', 'high', 'low']].copy()
    data['begin'] = pd.to_datetime(data['begin'])
    data['end'] = pd.to_datetime(data['end'])
    data['begin_date'] = data['begin'].dt.date
    data['begin_time'] = data['begin'].dt.time
    data['end_date'] = data['end'].dt.date
    data['end_time'] = data['end'].dt.time

    total_volume = data.groupby(by='begin_date')['volume'].agg('sum')
    data0 = data.merge(total_volume, on='begin_date')
    data0['volume_%'] = data0.volume_x / data0.volume_y
    data0['inventory_left'] = 1 - data0['volume_%']

    for i in range(1, len(data0)):
        if data0['volume_y'].iloc[i - 1] == data0['volume_y'].iloc[i]:
            data0.at[i, 'inventory_left'] = data0['inventory_left'].iloc[i - 1] - data0['volume_%'].iloc[i]

    data0['vwap'] = [
        np.sum(data0['close'].iloc[: i + 1] * data0['volume_x'].iloc[: i + 1]) / np.sum(data0['volume_x'].iloc[: i + 1])
        for i in range(len(data0))
    ]
    data0['av_price'] = 1 / 3 * (data0.low + data0.high + data0.close)

    revenue_table = evaluate_strategies(data0, x0=1)
    print(revenue_table)