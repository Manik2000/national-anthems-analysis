import os
import pickle

import numpy as np
import pandas as pd
import pycountry


def preprocess_lyrics(new_df):
    new_df["lyrics"] = ( 
    new_df["lyrics"]
    .str.replace("\n", " ")
    .str.replace("\r", " ")
    .str.replace("\t", " ")
    .str.replace(r" {2}", " ", regex=True)
    .str.strip()
    .str.replace(r"[0-9]*\.*", "", regex=True)
    .str.replace(r"chorus\:*", "", regex=True, case=False)
    )
    return new_df


if __name__ == "__main__":
    path = os.path.join("data", "data.pickle")

    with open(path, "rb") as handle:
        data = pickle.load(handle)

    _df = pd.DataFrame.from_dict(data, orient="index", columns=["iso_2", "lyrics"])
    countries_iso_2 = [country.alpha_2 for country in list(pycountry.countries)]
    countries_iso_3 = [country.alpha_3 for country in list(pycountry.countries)]

    df = _df.loc[_df["iso_2"].isin(countries_iso_2)].copy()  # Create a copy of the DataFrame slice

    df["iso_a3"] = np.array(countries_iso_3)[[countries_iso_2.index(i) for i in df["iso_2"].values]]
    df.loc["Kosovo", :] = _df.loc["Kosovo"].to_list() + ["-99"]
    df.loc["Somaliland", :] = _df.loc["Somaliland"].to_list() + ["SOL"]
    columns = ["iso_2", "iso_a3", "lyrics"]

    df = df.pipe(preprocess_lyrics)
    df = df[df["lyrics"] != ""]
    df.to_csv(os.path.join('data', 'for_embedding.csv'), index=False)
