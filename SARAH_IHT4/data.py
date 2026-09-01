import os
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np


def load_sp500(download=False):
    if download:
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(wiki_url)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"class": "wikitable"})
        sp500_constituents = []
        for row in table.find_all("tr")[1:]:
            symbol = row.find_all("td")[0].text.strip()
            sp500_constituents.append(symbol)
        sp500_constituents.append("^GSPC")
        total_data = []
        for symbol in sp500_constituents:
            constituent = yf.Ticker(symbol)
            data = constituent.history(start="2021-01-01", end="2022-12-31", interval="1d")
            total_data.append(data[['Open']].rename(columns={'Open': symbol}))
        total_df = pd.concat(total_data, axis=1)
        total_df.dropna(axis=1, inplace=True)
        total_df.to_csv(os.path.join("data", "sp500"))
        print("Data retrieval and saving completed.")
    dataset = pd.read_csv(os.path.join("data", "sp500"), index_col=0, low_memory=False)
    return dataset



def load_groups(download=False):
    if download:
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(wiki_url)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"class": "wikitable"})
        sp500_constituents = []
        sectors = []
        for row in table.find_all("tr")[1:]:
            symbol = row.find_all("td")[0].text.strip()
            sector = row.find_all("td")[2].text.strip()
            sp500_constituents.append(symbol)
            sectors.append(sector)
        sectors_joint_df = pd.concat([pd.Series(sp500_constituents), pd.Series(sectors)], axis=1)
        sectors_joint_df.to_csv(os.path.join("data", "sectors"))
    sectors = pd.read_csv(os.path.join("data", "sectors"), index_col=0, low_memory=False)
    return sectors


def build_group_idx(dataset, sectors):
    sectors_reindexed = sectors.set_index(['0'], inplace=False)
    sector_ids = np.unique(sectors['1'].values)
    cols = dataset.columns.values
    lookup_idx = dict()
    for idx in sector_ids:
        lookup_idx[idx] = []
    for (i, col) in enumerate(cols):
        if col !=  '^GSPC':
            lookup_idx[sectors_reindexed.loc[col].values[0]].append(i)
    return lookup_idx


def load_csi300(download=False):
    postfix = {'Shanghai': 'SS', 'Shenzhen': 'SZ'}
    if download:
        wiki_url = "https://en.wikipedia.org/wiki/CSI_300_Index"
        response = requests.get(wiki_url)
        soup = BeautifulSoup(response.text, "html.parser")
        constituents_section = soup.find("span", {"id": "Constituents"})
        if constituents_section:
            table = constituents_section.find_next("table", {"class": "wikitable"})
            if table:
                constituents = []
                sectors = []
                for row in table.find_all("tr")[1:]:
                    columns = row.find_all("td")
                    if len(columns) >= 3:
                        constituent = columns[0].get_text(strip=True)
                        pf = columns[2].get_text(strip=True)
                        constituents.append(constituent+f'.{postfix[pf]}')
        constituents.append("000300.SS")
        total_data = []
        for i, symbol in enumerate(constituents):
            constituent = yf.Ticker(symbol)
            data = constituent.history(start="2021-01-01", end="2022-12-31", interval="1d")
            total_data.append(data[['Open']].rename(columns={'Open': symbol}))
        data_of_index = total_data[-1]
        data_except_index = pd.concat(total_data[:-1], axis=1)
        data_except_index.dropna(axis=1, inplace=True)
        total_df = pd.merge(data_except_index, data_of_index, left_index=True, right_index=True, how='inner')
        total_df.to_csv(os.path.join("data", "csi300"))
        print("Data retrieval and saving completed.")
    dataset = pd.read_csv(os.path.join("data", "csi300"), index_col=0, low_memory=False)
    return dataset


def load_groups_csi300(download=False):
    postfix = {'Shanghai': 'SS', 'Shenzhen': 'SZ'}
    if download:
        constituents_url = "https://en.wikipedia.org/wiki/CSI_300_Index"
        response = requests.get(constituents_url)
        soup = BeautifulSoup(response.text, "html.parser")
        constituents_section = soup.find("span", {"id": "Constituents"})
        if constituents_section:
            table = constituents_section.find_next("table", {"class": "wikitable"})
            if table:
                constituents = []
                sectors = []
                for row in table.find_all("tr")[1:]:
                    columns = row.find_all("td")
                    if len(columns) >= 3:
                        constituent = columns[0].get_text(strip=True)
                        sector = columns[4].get_text(strip=True)
                        pf = columns[2].get_text(strip=True)
                        constituents.append(constituent+f'.{postfix[pf]}')
                        sectors.append(sector)
        sectors_joint_df = pd.concat([pd.Series(constituents), pd.Series(sectors)], axis=1)
        sectors_joint_df.to_csv(os.path.join("data", "sectors_csi300"))
    sectors = pd.read_csv(os.path.join("data", "sectors_csi300"), index_col=0, low_memory=False)
    return sectors


def build_group_idx_csi300(dataset, sectors):
    sectors_reindexed = sectors.set_index(['0'], inplace=False)
    sector_ids = np.unique(sectors['1'].values)
    cols = dataset.columns.values
    lookup_idx = dict()
    for idx in sector_ids:
        lookup_idx[idx] = []
    for (i, col) in enumerate(cols):
        if col !=  '000300.SS':
            lookup_idx[sectors_reindexed.loc[col].values[0]].append(i)
    return lookup_idx


def load_hsi(download=False):
    if download:
        wiki_url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
        response = requests.get(wiki_url)
        soup = BeautifulSoup(response.text, "html.parser")
        constituents_section = soup.find("span", {"id": "Components"})
        if constituents_section:
            table = constituents_section.find_next("table", {"class": "wikitable"})
            if table:
                constituents = []
                sectors = []
                for row in table.find_all("tr")[1:]:
                    columns = row.find_all("td")
                    if len(columns) >= 3:
                        constituent = columns[0].get_text(strip=True)
                        constituents.append(process_string_HK(constituent))
        constituents.append("^HSI")
        total_data = []
        for symbol in constituents:
            constituent = yf.Ticker(symbol)
            data = constituent.history(start="2021-01-01", end="2022-12-31", interval="1d")
            total_data.append(data[['Open']].rename(columns={'Open': symbol}))
        total_df = pd.concat(total_data, axis=1)
        total_df.dropna(axis=1, inplace=True)
        total_df.to_csv(os.path.join("data", "hsi"))
        print("Data retrieval and saving completed.")
    dataset = pd.read_csv(os.path.join("data", "hsi"), index_col=0, low_memory=False)
    return dataset


def load_groups_hsi(download=False):
    if download:
        constituents_url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
        response = requests.get(constituents_url)
        soup = BeautifulSoup(response.text, "html.parser")
        constituents_section = soup.find("span", {"id": "Components"})
        if constituents_section:
            table = constituents_section.find_next("table", {"class": "wikitable"})
            if table:
                constituents = []
                sectors = []
                for row in table.find_all("tr")[1:]:
                    columns = row.find_all("td")
                    constituent = columns[0].get_text(strip=True)
                    sector = columns[2].get_text(strip=True)
                    constituents.append(process_string_HK(constituent))
                    sectors.append(sector)
        sectors_joint_df = pd.concat([pd.Series(constituents), pd.Series(sectors)], axis=1)
        sectors_joint_df.to_csv(os.path.join("data", "sectors_hsi"))
    sectors = pd.read_csv(os.path.join("data", "sectors_hsi"), index_col=0, low_memory=False)
    return sectors


def build_group_idx_hsi(dataset, sectors):
    sectors_reindexed = sectors.set_index(['0'], inplace=False)
    sector_ids = np.unique(sectors['1'].values)
    cols = dataset.columns.values
    lookup_idx = dict()
    for idx in sector_ids:
        lookup_idx[idx] = []
    for (i, col) in enumerate(cols):
        if col !=  '^HSI':
            lookup_idx[sectors_reindexed.loc[col].values[0]].append(i)
    return lookup_idx


def pad(n):
    n_str = str(n)
    zeros_needed = 4 - len(n_str)
    padded_str = '0' * zeros_needed + n_str
    return padded_str

def process_string_HK(input_string):
    input_string = input_string[5:]
    padded_number = pad(input_string)
    result = f"{padded_number}.HK"
    return result
