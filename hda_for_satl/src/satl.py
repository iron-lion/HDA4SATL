import logging
from cdspp import *
from src.utils import *
from statistics import mean
from sklearn.model_selection import KFold
import multiprocessing as mp

# Set up logger for multiprocessing
globals()['logger'] = mp.get_logger()

# This class handles the testing of one dataset, testing all combinations of masked cells and preprocessing methods
class SATL:
    def __init__(self,
                 data_loader,
                 out_filename,
                 n_missing,
                 dim=11,
                 alpha=[0.01, 0.1, 1, 10, 100],
                 n_resample_source=300,
                 n_resample_target=300,
                 n_jobs=1,
                 K=10
                 ):
        """
        Constructor

        Args:
            n_missing (int): Number of masked labels (missing classes)
            dim (int): Dimension of the common latent space
            alpha (list): Regularization term values to test
            n_resample_source (int): Number of observations to sample for the source domain
            n_resample_target (int): Number of observations to sample for the target domain
            n_jobs (int): Number of parallel jobs to run
            K (int): Number of nearest neighbors for the semi-supervised model
        """
        self.data_loader = data_loader
        self.path_out = out_filename
        self.n_missing = n_missing
        self.dim = dim
        self.alpha = alpha
        self.n_source = n_resample_source
        self.n_target = n_resample_target
        self.n_jobs = n_jobs
        self.k = K

    def test_alpha(self, comb, alpha, mode):
        """
        Runs the model for the same combination with different alpha values.

        Args:
            comb (list): List of masked cells (missing classes)
            alpha (list): List of hyperparameters (alpha values) to test
            mode (str): Preprocessing mode to choose

        Returns:
            None. The results are saved to a CSV file.
        """
        X_source = self.data_loader['X_source']
        y_source = self.data_loader['y_source']

        X_train = self.data_loader['X_train']
        y_train = self.data_loader['y_train']

        X_test = self.data_loader['X_test']
        y_test = self.data_loader['y_test']

        # Up/Down sampling for balancing the datasets
        X_source, y_source = balance_sampling(X_source, y_source, self.n_source)
        X_train, y_train = balance_sampling(X_train, y_train, self.n_target)

        # Splitting the data into seen and unseen masked cells
        X_seen, X_unseen, y_seen, y_unseen = split_masked_cells(X_train, y_train, masked_cells=comb)
        logger.info(Counter(y_seen))

        df_dict = {"alpha": alpha}
        h_list = []
        acc_k_list = []
        acc_uk_list = []
        h_semi_list = []
        acc_k_semi_list = []
        acc_uk_semi_list = []
        for i in alpha:
            # Fit the model using the CDSPP algorithm (normal)
            model = CDSPP(X_source.T, y_source, i, self.dim, list(comb))
            model.fit(X_seen.T, y_seen)
            pred = model.predict(X_test.T)

            # Calculate performance metrics (H-score, accuracy for known and unknown classes)
            h, acc_known, acc_unknown = h_score(y_test, pred, comb)

            # Store the metrics in lists
            h_list.append(h)
            acc_k_list.append(acc_known)
            acc_uk_list.append(acc_unknown)

            # Fit the model using the semi-supervised CDSPP algorithm
            model.fit_semi_supervised(X_seen.T, X_test.T, y_seen)
            pred = model.predict(X_test.T)

            # Calculate performance metrics for the semi-supervised model
            h, acc_known, acc_unknown = h_score(y_test, pred, comb)
            h_semi_list.append(h)
            acc_k_semi_list.append(acc_known)
            acc_uk_semi_list.append(acc_unknown)

        # Create a DataFrame to store the results
        df_dict["H"] = h_list
        df_dict["Acc_known"] = acc_k_list
        df_dict["Acc_unknown"] = acc_uk_list
        df_dict["H_semi"] = h_semi_list
        df_dict["Acc_known_semi"] = acc_k_semi_list
        df_dict["Acc_unknown_semi"] = acc_uk_semi_list

        results = pd.DataFrame.from_dict(df_dict)
        results.to_csv(self.path_out + mode + "_alphas.csv")
        return


    def simple_feature_analysis(self, model, out_dir):
        source_fi = model.P_source.T @ self.data_loader['fi_source']
        source_fi = pd.DataFrame(source_fi)
        source_fi.columns = self.data_loader['id_source']
        source_fi = source_fi.T
        source_fi.to_csv(f'./results/{out_dir}_source_importance.csv', sep=',')

        target_fi = model.P_target.T @ self.data_loader['fi_target']
        target_fi = pd.DataFrame(target_fi)
        target_fi.columns = self.data_loader['id_target']
        target_fi = target_fi.T
        target_fi.to_csv(f'./results/{out_dir}_target_importance.csv', sep=',')


    def run_desc(self, missing_start, missing_end, mode):
        """
        Runs the model with an increasing share of masked cells (missing classes).

        Args:
            missing_start (int): Lowest number of missing classes
            missing_end (int): Highest number of missing classes
            mode (str): Preprocessing mode

        Returns:
            None. The results are saved to CSV files.
        """
        X_source = self.data_loader['X_source']
        y_source = self.data_loader['y_source']

        X_train = self.data_loader['X_train']
        y_train = self.data_loader['y_train']

        X_test = self.data_loader['X_test']
        y_test = self.data_loader['y_test']

        # Up/Down sampling for balancing the datasets
        X_source, y_source = balance_sampling(X_source, y_source, self.n_source)
        X_train, y_train = balance_sampling(X_train, y_train, self.n_target)

        for j in range(missing_start, missing_end + 1):
            # Extract possible combinations of masked cells
            self.n_missing = j
            logger.info(j)
            unique_classes = set(y_source)
            combs = list(combinations(unique_classes, self.n_missing))

            cols = []
            for i in range(self.n_missing):
                cols.append("Missing " + str(i + 1))
            cols = cols + ["alpha", "y_true", "y_pred", "y_pred_semi"]
            results = pd.DataFrame(columns=cols)

            # Each combination is run on a different core (parallel processing)
            arguments = [(X_source, X_train, X_test, y_source, y_train, y_test, list(i), cols) for i in combs]
            pool = mp.Pool(processes=self.n_jobs)
            all_results = pool.starmap(self.run_combination, arguments)
            pool.close()

            for result in all_results:
                results = pd.concat((results, result), ignore_index=True)
            results.to_csv('./results/' + self.path_out + mode + "_pred.csv")
            scores = get_all_for_desc(results, combs, True, self.path_out + str(j) + "_" + mode + "_h.csv")
        return

    def run_mode(self, pseudo=True):
        """
        Runs all combinations of masked cells and preprocessing methods.

        Args:
            pseudo (bool): Flag indicating whether to use pseudo-labeling (default: True)

        Returns:
            None. The results are saved to a CSV file.
        """
        X_source = self.data_loader['X_source']
        y_source = self.data_loader['y_source']

        X_train = self.data_loader['X_train']
        y_train = self.data_loader['y_train']

        X_test = self.data_loader['X_test']
        y_test = self.data_loader['y_test']

        # Up/Down sampling for balancing the datasets
        if self.n_source > 0:
            X_source, y_source = balance_sampling(X_source, y_source, self.n_source)
        if self.n_target > 0:
            X_train, y_train = balance_sampling(X_train, y_train, self.n_target)

        # Extract possible combinations of masked cells
        unique_classes = set(y_source)
        combs = list(combinations(unique_classes, self.n_missing))

        # Prepare DataFrame to store the results
        cols = []
        for i in range(self.n_missing):
            cols.append("Missing " + str(i + 1))
        cols = cols + ["alpha", "y_true", "y_pred", "y_pred_semi"]
        results = pd.DataFrame(columns=cols)

        # Each combination is run on a different core (parallel processing)
        arguments = [(X_source, X_train, X_test, y_source, y_train, y_test, list(i), cols, pseudo) for i in combs]
        all_results = []

        if self.n_jobs > 1:
            with mp.Pool(processes=self.n_jobs) as pool:
                for result in pool.starmap(self.run_combination, arguments):
                    all_results.append(result)
        else:
            for args in arguments:
                x = (self.run_combination(*args))
                all_results.append(x)

        for result in all_results:
            results = pd.concat((results, result), ignore_index=True)
        results.to_csv('./results/' + self.path_out + "_pred.csv")
        scores = get_all(results, True, './results/' + self.path_out + "_h.csv")
        return

    def run_combination(self, X_source, X_train, X_test, y_source, y_train, y_test, comb, cols, pseudo):
        """
        Runs one combination of masked cells (missing classes).

        Args:
            X_source (array-like): Source domain features
            X_train (array-like): Target domain training features
            X_test (array-like): Target domain test features
            y_source (array-like): Source domain labels
            y_train (array-like): Target domain training labels
            y_test (array-like): Target domain test labels
            comb (list): List of masked cells (missing classes)
            cols (list): Column names of the final DataFrame
            pseudo (bool): Flag indicating whether to use pseudo-labeling

        Returns:
            DataFrame: DataFrame with results
        """
        # Mask combinations to be tested
        X_seen, X_unseen, y_seen, y_unseen = split_masked_cells(X_train, y_train, masked_cells=comb)
        logger.info("Masked cells: " + str(comb))

        # Cross-validate the regularization alpha
        alpha = self.get_alpha(X_source, X_seen, y_source, y_seen) if len(self.alpha) > 1 else self.alpha
        logger.info("Best alpha: " + str(alpha))

        # Fit the model using the CDSPP algorithm (normal)
        model = CDSPP(X_source.T, y_source, alpha, self.dim, list(comb))
        model.fit(X_seen.T, y_seen)
        pred = model.predict(X_test.T)
        
        z_source = model.transform_source()
        z_target = model.transform_target(X_test.T)
        plot_latent(z_source, y_source, z_target, y_test, pred, comb, f'{self.path_out}_{"-".join([str(x) for x in comb])}.png')
        self.simple_feature_analysis(model, f'{self.path_out}_{"-".join([str(x) for x in comb])}')
        logger.info(f'...{comb} Fit done')

        if pseudo:
            # Fit the model using the semi-supervised CDSPP algorithm
            model.fit_semi_supervised(X_seen.T, X_test.T, y_seen, K_seen=self.k, K_unseen=self.k)
            pred_semi = model.predict(X_test.T)
            z_source = model.transform_source()
            z_target = model.transform_target(X_test.T)
            plot_latent(z_source, y_source, z_target, y_test, pred_semi, comb, f'{self.path_out}_{"-".join([str(x) for x in comb])}_pseudo.png')
            self.simple_feature_analysis(model, f'{self.path_out}_{"-".join([str(x) for x in comb])}')
            logger.info(f'...{comb} Fit_pseudo done')
        else:
            pred_semi = -1

        n_test = len(y_test)
        results_dict = dict()
        for i in range(len(cols) - 4):
            results_dict[cols[i]] = n_test * [comb[i]]
        results_dict["alpha"] = n_test * [alpha]
        results_dict["y_true"] = y_test
        results_dict["y_pred"] = pred
        results_dict["y_pred_semi"] = pred_semi
        return pd.DataFrame.from_dict(results_dict)

    def get_alpha(self, X_source, X, y_source, y):
        """
        Performs sample- and class-wise cross-validation to determine the best alpha value.

        Args:
            X_source (array-like): Source domain features
            X (array-like): Target domain features (train)
            y_source (array-like): Source domain labels
            y (array-like): Target domain labels (train)

        Returns:
            float: Best alpha value
        """
        overall_score = []
        for i in range(len(self.alpha)):
            score = []
            for train_index, test_index in KFold(shuffle=True, n_splits=5).split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                for j in set(y):
                    X_seen, X_unseen, y_seen, y_unseen = split_masked_cells(X_train, y_train, masked_cells=[j])
                    model = CDSPP(X_source.T, y_source, self.alpha[i], self.dim)
                    model.fit(X_seen.T, y_seen)
                    pred = model.predict(X_test.T)
                    h, acc_known, acc_unknown = h_score(y_test, pred, [j])
                    score.append(acc_unknown)
            overall_score.append(mean(score))
        return self.alpha[np.argmax(np.array(overall_score))]
