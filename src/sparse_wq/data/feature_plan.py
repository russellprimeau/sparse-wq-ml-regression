"""Native observation memberships and feature layout for repeated subset evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sparse_wq.data.native import DEEP_SUFFIX, block_edges

class NativeFeaturePlan:

    def __init__(self, observations, features, start, cutoff, representation):
        self.features = list(features)
        self.representation = representation
        self.deep = np.array([name.endswith(DEEP_SUFFIX) for name in features])
        self.original = observations[self.features].to_numpy(dtype=float)
        self.finite = np.isfinite(self.original)
        index = observations.index
        times = pd.DatetimeIndex(index.get_level_values('TIMESTAMP') if isinstance(index, pd.MultiIndex) else index)
        self.metadata = []
        if representation == 'none':
            if self.deep.any():
                raise ValueError('No hourly conversion for deep series')
            hours = pd.date_range(start, cutoff, freq='h')
            self.positions = np.full((len(hours), len(features)), -1, dtype=int)
            source = index.get_level_values('series').to_numpy() if isinstance(index, pd.MultiIndex) else np.repeat('source', len(times))
            records = index.get_level_values('RECORD').to_numpy() if isinstance(index, pd.MultiIndex) and 'RECORD' in index.names else np.arange(len(times))
            labels = times.ceil('h')
            bins = hours.get_indexer(labels)
            chosen = {}
            for position in np.lexsort((np.arange(len(times)), records, times.as_unit('ns').asi8)):
                if bins[position] >= 0:
                    chosen[source[position], bins[position]] = position
            for column, name in enumerate(features):
                supplied = np.unique(source[self.finite[:, column]])
                if len(supplied) > 1:
                    raise ValueError('Multiple sources for one predictor')
                if len(supplied):
                    for hour in range(len(hours)):
                        self.positions[hour, column] = chosen.get((supplied[0], hour), -1)
            for hour in range(len(hours)):
                for name in features:
                    self.metadata.append(dict(column=name, scope='hour', block_index=hour, statistic='value'))
            return
        edges = block_edges(start, cutoff, representation)
        ids = np.searchsorted(edges.as_unit('ns').asi8, times.as_unit('ns').asi8, side='right') - 1
        self.groups = [np.flatnonzero(ids == block) for block in range(len(edges) - 1)]
        self.window = np.flatnonzero((ids >= 0) & (ids < len(edges) - 1))
        self.counts = [self.finite[group].sum(axis=0) for group in self.groups]
        self.window_counts = self.finite[self.window].sum(axis=0)
        for column, name in enumerate(features):
            for block in range(len(self.groups)):
                statistics = ['mean'] if representation == 'mean' else ['mean', 'min', 'max'] + ([] if self.deep[column] else ['std'])
                for statistic in statistics:
                    self.metadata.append(dict(column=name, scope='block', block_index=block, statistic=statistic))
            if self.deep[column] and representation != 'mean':
                self.metadata.append(dict(column=name, scope='window', block_index=-1, statistic='std'))

    @staticmethod
    def moments(values, count):
        finite = np.isfinite(values)
        mean = np.divide(np.where(finite, values, 0.0).sum(axis=0), count, out=np.full(values.shape[1], np.nan), where=count > 0)
        variance = np.divide(np.where(finite, (values - mean) ** 2, 0.0).sum(axis=0), count, out=np.full(values.shape[1], np.nan), where=count > 0)
        return (mean, np.sqrt(variance))

    def encode(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != self.original.shape or not np.array_equal(np.isfinite(values), self.finite):
            raise ValueError('Scaling must preserve observation identity and missingness')
        if self.representation == 'none':
            result = np.full(self.positions.shape, np.nan)
            valid = self.positions >= 0
            columns = np.broadcast_to(np.arange(len(self.features)), self.positions.shape)
            result[valid] = values[self.positions[valid], columns[valid]]
            return result.flatten()
        blocks = []
        for group, count in zip(self.groups, self.counts):
            chunk = values[group]
            mean, std = self.moments(chunk, count)
            sufficient = count >= np.where(self.deep, 1, 2)
            if self.representation == 'mean':
                blocks.append(dict(mean=np.where(sufficient, mean, np.nan)))
                continue
            minimum = np.where(np.isfinite(chunk), chunk, np.inf).min(axis=0) if len(chunk) else np.full(len(self.features), np.inf)
            maximum = np.where(np.isfinite(chunk), chunk, -np.inf).max(axis=0) if len(chunk) else np.full(len(self.features), -np.inf)
            blocks.append(dict(mean=np.where(sufficient, mean, np.nan), min=np.where(sufficient, minimum, np.nan), max=np.where(sufficient, maximum, np.nan), std=np.where(count >= 2, std, np.nan)))
        _, window_std = self.moments(values[self.window], self.window_counts)
        return np.array([window_std[self.features.index(row['column'])] if row['scope'] == 'window' and self.window_counts[self.features.index(row['column'])] >= 2 else np.nan if row['scope'] == 'window' else blocks[row['block_index']][row['statistic']][self.features.index(row['column'])] for row in self.metadata])
