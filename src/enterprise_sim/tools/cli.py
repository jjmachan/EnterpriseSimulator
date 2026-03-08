"""Main CLI entry point for esim."""

import click

from enterprise_sim.tools.employee_tools import (
    check_order,
    lookup_customer,
    send_reply,
    update_ticket,
)


@click.group()
def cli():
    """EnterpriseSim CLI tools."""
    pass


cli.add_command(lookup_customer)
cli.add_command(check_order)
cli.add_command(send_reply)
cli.add_command(update_ticket)


if __name__ == "__main__":
    cli()
