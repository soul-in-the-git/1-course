import numpy as np
import pandas as pd


def split_date(cycling: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(cycling["timestamp"])
    parts = pd.DataFrame(
        {
            "Year": date.dt.year,
            "Month": date.dt.month,
            "Day": date.dt.day,
        }
    )
    return pd.concat([parts, cycling.drop(columns="timestamp")], axis=1)


def prepare(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # делаем числа в date
    new_train = split_date(train)
    new_test = split_date(test)

    # считаем значения для пропусков
    new_train_mean_life = new_train["life_sq"].mean()
    new_train_mean_metro = new_train["metro_km_walk"].mean()
    new_train_mean_kitch = new_train["kitch_sq"].mean()
    new_train_median_floor = new_train["floor"].median()
    new_train_mean_max_floor = new_train["max_floor"].median()
    new_train_median_build_year = new_train["build_year"].median()
    new_train_median_num_room = new_train["num_room"].median()
    new_train_median_mater = new_train["material"].median()
    new_train_median_state = new_train["state"].median()

    # заполняем
    new_train["life_sq"] = new_train["life_sq"].fillna(new_train_mean_life)
    new_train["metro_km_walk"] = new_train["metro_km_walk"].fillna(new_train_mean_metro)
    new_train["kitch_sq"] = new_train["kitch_sq"].fillna(new_train_mean_kitch)
    new_train["floor"] = new_train["floor"].fillna(new_train_median_floor)
    new_train["max_floor"] = new_train["max_floor"].fillna(new_train_mean_max_floor)
    new_train["build_year"] = new_train["build_year"].fillna(
        new_train_median_build_year
    )
    new_train["num_room"] = new_train["num_room"].fillna(new_train_median_num_room)
    new_train["material"] = new_train["material"].fillna(new_train_median_mater)
    new_train["state"] = new_train["state"].fillna(new_train_median_state)
    train_without_na = new_train.copy()

    new_test["life_sq"] = new_test["life_sq"].fillna(new_train_mean_life)
    new_test["metro_km_walk"] = new_test["metro_km_walk"].fillna(new_train_mean_metro)
    new_test["kitch_sq"] = new_test["kitch_sq"].fillna(new_train_mean_kitch)
    new_test["floor"] = new_test["floor"].fillna(new_train_median_floor)
    new_test["max_floor"] = new_test["max_floor"].fillna(new_train_mean_max_floor)
    new_test["build_year"] = new_test["build_year"].fillna(new_train_median_build_year)
    new_test["num_room"] = new_test["num_room"].fillna(new_train_median_num_room)
    new_test["material"] = new_test["material"].fillna(new_train_median_mater)
    new_test["state"] = new_test["state"].fillna(new_train_median_state)
    test_without_na = new_test.copy()

    # преобразуем строки в числа (one-hot encoding): тип продукта и жд
    train_without_na["Invest"] = np.where(
        train_without_na["product_type"] == "Investment", 1, 0
    )
    train_without_na["Own"] = np.where(
        train_without_na["product_type"] == "OwnerOccupier", 1, 0
    )
    train_without_na = train_without_na.drop(columns="product_type")

    test_without_na["Invest"] = np.where(
        test_without_na["product_type"] == "Investment", 1, 0
    )
    test_without_na["Own"] = np.where(
        test_without_na["product_type"] == "OwnerOccupier", 1, 0
    )
    test_without_na = test_without_na.drop(columns="product_type")

    train_without_na["railroad_1line_yes"] = np.where(
        train_without_na["railroad_1line"] == "yes", 1, 0
    )
    train_without_na["railroad_1line_no"] = np.where(
        train_without_na["railroad_1line"] == "no", 1, 0
    )
    train_1 = train_without_na.drop(columns="railroad_1line")

    test_without_na["railroad_1line_yes"] = np.where(
        test_without_na["railroad_1line"] == "yes", 1, 0
    )
    test_without_na["railroad_1line_no"] = np.where(
        test_without_na["railroad_1line"] == "no", 1, 0
    )
    test_1 = test_without_na.drop(columns="railroad_1line")

    # школа
    train_1["child_on_acc_pre_school"] = train_1["child_on_acc_pre_school"].str.replace(
        ",", "."
    )
    test_1["child_on_acc_pre_school"] = test_1["child_on_acc_pre_school"].str.replace(
        ",", "."
    )
    train_1["child_on_acc_pre_school"] = train_1["child_on_acc_pre_school"].replace(
        "#!", "0.0"
    )
    test_1["child_on_acc_pre_school"] = test_1["child_on_acc_pre_school"].replace(
        "#!", "0.0"
    )
    train_1["child_on_acc_pre_school"] = train_1["child_on_acc_pre_school"].map(
        lambda x: float(x)
    )
    test_1["child_on_acc_pre_school"] = test_1["child_on_acc_pre_school"].map(
        lambda x: float(x)
    )
    train_1["child_on_acc_pre_school"] = train_1["child_on_acc_pre_school"].fillna(0.0)
    test_1["child_on_acc_pre_school"] = test_1["child_on_acc_pre_school"].fillna(0.0)
    train_2 = train_1.copy()
    test_2 = test_1.copy()

    # Экология
    train_2["ecology"] = train_2["ecology"].replace("poor", "1")
    train_2["ecology"] = train_2["ecology"].replace("no data", "0")
    train_2["ecology"] = train_2["ecology"].replace("satisfactory", "2")
    train_2["ecology"] = train_2["ecology"].replace("good", "3")
    train_2["ecology"] = train_2["ecology"].replace("excellent", "4")
    train_2["ecology"] = train_2["ecology"].map(lambda x: int(x))
    train_3 = train_2.copy()

    test_2["ecology"] = test_2["ecology"].replace("poor", "1")
    test_2["ecology"] = test_2["ecology"].replace("no data", "0")
    test_2["ecology"] = test_2["ecology"].replace("satisfactory", "2")
    test_2["ecology"] = test_2["ecology"].replace("good", "3")
    test_2["ecology"] = test_2["ecology"].replace("excellent", "4")
    test_2["ecology"] = test_2["ecology"].map(lambda x: int(x))
    test_3 = test_2.copy()

    # районы по частоте
    train_sub = train_3["sub_area"].value_counts() / np.size(train_3["sub_area"])
    train_3["sub_area"] = train_3["sub_area"].map(train_sub)
    test_3["sub_area"] = test_3["sub_area"].map(train_sub)

    return train_3, test_3


if __name__ == "__main__":
    train = pd.read_csv("train.csv").drop(columns="price_doc")
    prepared, _ = prepare(train, train.head(50).copy())
    prepared.head(50).to_csv("output.csv", index=False)
