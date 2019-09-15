import asyncio
from pathlib import Path

import click

from deepspain.dataset import download, generate_ids


@click.command()
@click.argument("from_boe_id", metavar="<boe-id>")
@click.argument(
    "output_file", metavar="<output.jsonlines>", type=click.Path(dir_okay=False)
)
def main(from_boe_id: str, output_file: str):
    """Fetches all BOEs starting at <boe-id> until yesterday's, and stores them in <output.jsonlines>.
    Running multiple times will append to that file."""
    boe_ids = list(generate_ids(from_boe_id))
    if Path(output_file).exists():
        click.echo(output_file + " already exists, so we'll be adding to it.")
    click.echo("Fetching " + str(len(boe_ids)) + " since " + from_boe_id + "✨")
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(download(boe_ids, output_file))
    loop.run_until_complete(future)
    click.echo("Fetched " + str(len(boe_ids)) + " items.")


if __name__ == "__main__":
    main()
