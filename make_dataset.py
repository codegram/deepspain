import click
import json
import re
import io
from aiohttp import ClientSession
import xml.etree.ElementTree as ET
import asyncio
from datetime import date, timedelta

base_url = "https://www.boe.es"


def parse_item(item, extra, content):
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
            [x.text for x in texto.findall('p[@class="firma_ministro"]')]
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


async def fetch_item(item, session, metadata):
    url = base_url + item.findtext("urlXml")
    res = await session.request(method="GET", url=url)
    text = await res.text()
    parsed = parse_item(item, metadata, text)

    return parsed


async def fetch(url, session):
    async with session.get(url) as response:
        return await response.read()


async def fetch_boe(session, id):
    url = base_url + "/diario_boe/xml.php?id=" + id

    resp = await fetch(url, session)

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
                            asyncio.ensure_future(
                                fetch_item(item, session, metadata))
                            for item in epigraph.findall("./item")
                        ]
    except Exception:
        pass

    return await asyncio.gather(*futures)


async def run(boe_ids):
    with io.open("output.jsonlines", "a", encoding="utf8") as json_file:
        n = 1
        for boe_id in boe_ids:
            print(str(n) + "/" + str(len(boe_ids)) + " - " + boe_id)
            n = n + 1
            async with ClientSession() as session:
                items = await fetch_boe(session, boe_id)
                for item in items:
                    json.dump(item, json_file, ensure_ascii=False)
                    json_file.write("\n")


one_day = timedelta(days=1)


def generate_boe_ids(from_boe: str):
    year, month, day = re.search(r"(\d{4})(\d{2})(\d{2})", from_boe).groups()
    start = date(int(year), int(month), int(day))
    yesterday = date.today() - one_day
    d = start - one_day
    while d < yesterday:
        d = d + one_day
        yield "BOE-S-" + d.strftime("%Y%m%d")


@click.command()
@click.argument("from_boe_id")
@click.argument("output_file", type=click.Path(dir_okay=False))
def main(from_boe_id: str, output_file: str):
    boe_ids = list(generate_boe_ids(from_boe_id))
    if output_file.exists():
        click.echo(output_file + " already exists, so we'll be adding to it.")
    click.echo("Fetching " + str(len(boe_ids)) + " since " + from_boe_id + "✨")
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(run(boe_ids))
    items = loop.run_until_complete(future)
    click.echo("Fetched " + str(len(items)) + " items.")


if __name__ == "__main__":
    main()
