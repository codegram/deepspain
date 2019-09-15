"""Handles creation and processing of the BOE  dataset from scratch."""

import asyncio
import io
import json
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from aiohttp import ClientSession
from fastai.text.data import TextList, TextLMDataBunch, load_data
from pandas import DataFrame

from deepspain.utils import measure

BASE_URL = "https://www.boe.es"


def _parse_item(item, extra, content):
    item_id = item.get("id")
    item_control = item.get("control")
    title = item.findtext("titulo")
    url = item.findtext("urlXml")

    try:
        root = ET.fromstring(content)

        metadata = root.find("metadatos")
        date = metadata.findtext("fecha_disposicion")
        rank = metadata.findtext("rango")
        publish_date = metadata.findtext("fecha_publicacion")
        texto = root.find("texto")

        content = (
            "\n".join([x.text for x in texto.findall("p") if x.text])
            .replace("\xa0", " ")
            .replace("\u2003", " ")
        )
        signatures = " ".join(
            [x.text for x in texto.findall('p[@class="firma_ministro"]') if x.text]
        )
        result = dict(
            id=item_id,
            rank=rank,
            control=item_control,
            title=title,
            url=url,
            content=content,
            signatures=signatures,
            date=date,
            publish_date=publish_date,
            **extra
        )
        return result
    except Exception:
        return None


async def _fetch_item(item, session, metadata):
    url = BASE_URL + item.findtext("urlXml")
    res = await session.request(method="GET", url=url)
    text = await res.text()
    parsed = _parse_item(item, metadata, text)

    return parsed


async def _fetch(url, session):
    async with session.get(url) as response:
        return await response.read()


async def _fetch_boe(session, id):
    url = BASE_URL + "/diario_boe/xml.php?id=" + id

    resp = await _fetch(url, session)

    futures = []
    try:
        root = ET.fromstring(resp)

        if not root.find("./error"):
            for section in root.findall("./diario/seccion"):
                attrs = section.attrib
                num = attrs["num"]
                name = attrs["nombre"]
                for department in section.findall("./departamento"):
                    department_name = department.attrib["nombre"]
                    for epigraph in department.findall("./epigrafe"):
                        epigraph_name = epigraph.attrib["nombre"]
                        metadata = dict(
                            boe_id=id,
                            section_number=num,
                            section_name=name,
                            department_name=department_name,
                            epigraph_name=epigraph_name,
                        )
                        futures = futures + [
                            asyncio.ensure_future(_fetch_item(item, session, metadata))
                            for item in epigraph.findall("./item")
                        ]
    except Exception:
        pass

    return await asyncio.gather(*futures)


async def download(boe_ids: Sequence[str], output_file: str):
    """Downloads a series of `boe ids`, saving them to `output_file` as a
    JSONlines file. Appends to an existing file if needed."""

    with io.open(output_file, "a", encoding="utf8") as json_file:
        n = 1
        for boe_id in boe_ids:
            print(str(n) + "/" + str(len(boe_ids)) + " - " + boe_id)
            n = n + 1
            async with ClientSession() as session:
                items = await _fetch_boe(session, boe_id)
                for item in items:
                    json.dump(item, json_file, ensure_ascii=False)
                    json_file.write("\n")


one_day = timedelta(days=1)


def generate_ids(from_boe: str) -> Iterator[str]:
    """Returns an iterator over BOE ids from a specific starting id until
    yesterday's."""

    year, month, day = re.search(r"(\d{4})(\d{2})(\d{2})", from_boe).groups()
    start = date(int(year), int(month), int(day))
    yesterday = date.today() - one_day
    d = start - one_day
    while d < yesterday:
        d = d + one_day
        yield "BOE-S-" + d.strftime("%Y%m%d")


def df_to_lm_databunch(
    df: DataFrame,
    columns: Sequence[str],
    batch_size: int = 48,
    seed: int = 42,
    sample: bool = False,
) -> TextLMDataBunch:
    """Extracts text from `columns` in `df` and produces a DataBunch ready for language modeling."""
    np.random.seed(seed)

    data = TextList.from_df(df, cols=columns)
    if sample:
        data = data.filter_by_rand(0.01, seed=seed)

    databunch = (
        data.split_by_rand_pct(0.2, seed=seed).label_for_lm().databunch(bs=batch_size)
    )
    return databunch


def load_databunch(pkl_path: Path, debug=False) -> TextLMDataBunch:
    p = pkl_path
    folder = p.parent
    filename = p.name
    return measure("loading dataframe", lambda: load_data(folder, filename), debug)
