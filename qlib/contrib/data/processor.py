import numpy as np

from ...log import TimeInspector
from ...data.dataset.processor import Processor, get_group_columns


class ConfigSectionProcessor(Processor):
    """
    This processor is designed for Alpha158. And will be replaced by simple processors in the future
    """

    def __init__(self, fields_group=None, **kwargs):
        super().__init__()
        # Options
        self.fillna_feature = kwargs.get("fillna_feature", True)
        self.fillna_label = kwargs.get("fillna_label", True)
        self.clip_feature_outlier = kwargs.get("clip_feature_outlier", False)
        self.shrink_feature_outlier = kwargs.get("shrink_feature_outlier", True)
        self.clip_label_outlier = kwargs.get("clip_label_outlier", False)

        self.fields_group = None

    def __call__(self, df):
        return self._transform(df)

    def _transform(self, df):
        def _label_norm(x):
            x = x - x.mean()  # copy
            x /= x.std()
            if self.clip_label_outlier:
                x.clip(-3, 3, inplace=True)
            if self.fillna_label:
                x.fillna(0, inplace=True)
            return x

        def _feature_norm(x):
            x = x - x.median()  # copy
            x /= x.abs().median() * 1.4826
            if self.clip_feature_outlier:
                x.clip(-3, 3, inplace=True)
            if self.shrink_feature_outlier:
                x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5, inplace=True)
                x.where(x >= -3, -3 - (x + 3).div(x.min() + 3) * 0.5, inplace=True)
            if self.fillna_feature:
                x.fillna(0, inplace=True)
            return x

        TimeInspector.set_time_mark()

        # Copy the focus part and change it to single level
        selected_cols = get_group_columns(df, self.fields_group)
        df_focus = df[selected_cols].copy()
        if len(df_focus.columns.levels) > 1:
            df_focus = df_focus.droplevel(level=0)

        # Label
        cols = df_focus.columns[df_focus.columns.str.contains("^LABEL")]
        df_focus[cols] = df_focus[cols].groupby(level="datetime", group_keys=False).apply(_label_norm)

        # Features
        cols = df_focus.columns[df_focus.columns.str.contains("^KLEN|^KLOW|^KUP")]
        df_focus[cols] = (
            df_focus[cols].apply(lambda x: x**0.25).groupby(level="datetime", group_keys=False).apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^KLOW2|^KUP2")]
        df_focus[cols] = (
            df_focus[cols].apply(lambda x: x**0.5).groupby(level="datetime", group_keys=False).apply(_feature_norm)
        )

        _cols = [
            "KMID",
            "KSFT",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VWAP",
            "ROC",
            "MA",
            "BETA",
            "RESI",
            "QTLU",
            "QTLD",
            "RSV",
            "SUMP",
            "SUMN",
            "SUMD",
            "VSUMP",
            "VSUMN",
            "VSUMD",
        ]
        pat = "|".join(["^" + x for x in _cols])
        cols = df_focus.columns[df_focus.columns.str.contains(pat) & (~df_focus.columns.isin(["HIGH0", "LOW0"]))]
        df_focus[cols] = df_focus[cols].groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")]
        df_focus[cols] = df_focus[cols].apply(np.log).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^RSQR")]
        df_focus[cols] = df_focus[cols].fillna(0).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^MAX|^HIGH0")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: (x - 1) ** 0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^MIN|^LOW0")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: (1 - x) ** 0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^CORR|^CORD")]
        df_focus[cols] = df_focus[cols].apply(np.exp).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^WVMA")]
        df_focus[cols] = df_focus[cols].apply(np.log1p).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        df[selected_cols] = df_focus.values

        TimeInspector.log_cost_time("Finished preprocessing data.")

        return df


class CSNeutralize(Processor):
    """
    Cross Sectional Neutralization Processor.
    
    对 feature 和 label 进行市值中性化和/或行业中性化处理。
    通过回归残差法实现：y = alpha + beta * X + epsilon，返回 epsilon 作为中性化后的值。
    
    Parameters
    ----------
    fields_group : list or str
        要中性化的字段组，如 ["feature", "label"]
    market_cap_col : str, optional
        市值列名（在 feature 中），默认 "total_mv"。设为 None 则不做市值中性化。
    industry_col : str, optional
        行业列名（在 feature 中），默认 "industry_id"。设为 None 则不做行业中性化。
    use_log_cap : bool
        是否对市值取 log，默认 True
    """

    def __init__(
        self,
        fields_group=None,
        market_cap_col="total_mv",
        industry_col="industry_id",
        use_log_cap=True,
    ):
        super().__init__()
        if fields_group is None:
            self.fields_group = ["feature", "label"]
        elif not isinstance(fields_group, list):
            self.fields_group = [fields_group]
        else:
            self.fields_group = fields_group
        self.market_cap_col = market_cap_col
        self.industry_col = industry_col
        self.use_log_cap = use_log_cap

    def _neutralize_group(self, group_df, target_cols):
        """对单个截面进行中性化处理"""
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        
        n_samples = len(group_df)
        if n_samples < 3:
            return group_df
        
        # 构建回归自变量
        X_parts = []
        
        # 市值因子
        if self.market_cap_col is not None:
            cap_col = ("feature", self.market_cap_col)
            if cap_col in group_df.columns:
                cap_values = group_df[cap_col].values.reshape(-1, 1)
                if self.use_log_cap:
                    cap_values = np.log1p(np.abs(cap_values))
                X_parts.append(cap_values)
        
        # 行业哑变量
        if self.industry_col is not None:
            ind_col = ("feature", self.industry_col)
            if ind_col in group_df.columns:
                industry_values = group_df[ind_col].fillna(-1).astype(int)
                unique_industries = industry_values.unique()
                if len(unique_industries) > 1:
                    dummies = pd.get_dummies(industry_values, drop_first=True, dtype=float)
                    X_parts.append(dummies.values)
        
        if not X_parts:
            return group_df
        
        X = np.hstack(X_parts)
        valid_mask = ~np.isnan(X).any(axis=1)
        if valid_mask.sum() < 3:
            return group_df
        
        result = group_df.copy()
        reg = LinearRegression()
        
        for col in target_cols:
            if col not in group_df.columns:
                continue
            y = group_df[col].values
            col_valid = valid_mask & ~np.isnan(y)
            
            if col_valid.sum() < 3:
                continue
            
            reg.fit(X[col_valid], y[col_valid])
            
            # Only predict for rows with valid X
            # Note: valid_mask guarantees X is not NaN
            y_pred = reg.predict(X[valid_mask])
            
            # Calculate residual for valid rows
            # y[valid_mask] might contain NaNs, so residual will be NaN there, which is correct
            residual = y[valid_mask] - y_pred
            
            # Assign residuals back to result
            # Use columns' values to avoid index alignment issues if possible, 
            # but here we need to align with original dataframe index.
            result.loc[result.index[valid_mask], col] = residual
        
        return result

    def __call__(self, df):
        import pandas as pd
        
        target_cols = []
        for g in self.fields_group:
            if g == "feature":
                exclude = {self.market_cap_col, self.industry_col}
                cols = get_group_columns(df, g)
                for c in cols:
                    col_name = c[1] if isinstance(c, tuple) else c
                    if col_name not in exclude:
                        target_cols.append(c)
            else:
                cols = get_group_columns(df, g)
                target_cols.extend(cols)
        
        if not target_cols:
            return df
        
        with pd.option_context("mode.chained_assignment", None):
            df = df.groupby("datetime", group_keys=False).apply(
                lambda x: self._neutralize_group(x, target_cols)
            )
        
        return df

