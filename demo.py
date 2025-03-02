import numpy as np
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from joblib import parallel_backend

# 1️⃣ Tạo dữ liệu giả lập
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2️⃣ Khởi tạo mô hình
model = RandomForestClassifier(random_state=42)

# 3️⃣ Định nghĩa các tham số tìm kiếm
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
}


# 4️⃣ Tạo tqdm để hiển thị tiến trình GridSearchCV
def tqdm_joblib(tqdm_object):
    """Tạo thanh tiến trình với joblib"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_parallel = joblib.Parallel

    def new_parallel(*args, **kwargs):
        kwargs["batch_completion_callback"] = TqdmBatchCompletionCallback(tqdm_object)
        return old_parallel(*args, **kwargs)

    joblib.Parallel = new_parallel
    return tqdm_object


# 5️⃣ Chạy GridSearchCV với tqdm
if __name__ == "__main__":
    with parallel_backend("threading"):
        with tqdm_joblib(
            tqdm(
                desc="Đang chạy GridSearch",
                total=len(param_grid["n_estimators"]) * len(param_grid["max_depth"]),
            )
        ) as progress_bar:
            grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=0)
            grid_search.fit(X_train, y_train)

    # 6️⃣ Hiển thị kết quả tốt nhất
    print("\n🏆 Best Parameters:", grid_search.best_params_)

df_filled[ordinal_cols] = df_filled[ordinal_cols].astype("category")
