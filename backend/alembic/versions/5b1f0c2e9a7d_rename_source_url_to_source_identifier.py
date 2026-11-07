'''rename molecule_sources.source_url to source_identifier

Store the provider's identifier (e.g. CHEMBL25) instead of a URL; links are built at request time.

Revision ID: 5b1f0c2e9a7d
Revises: 08d3e13077d6
Create Date: 2026-11-06 15:00:00.000000

'''

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5b1f0c2e9a7d"
down_revision: Union[str, Sequence[str], None] = "08d3e13077d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("molecule_sources", "source_url", new_column_name="source_identifier")


def downgrade() -> None:
    op.alter_column("molecule_sources", "source_identifier", new_column_name="source_url")
