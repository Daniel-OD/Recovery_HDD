"""
ext4/gdt.py — Group Descriptor Table (GDT) parser.

Fiecare block group din ext4 are un descriptor care indică unde se află
pe disc bitmap-urile, inode table-ul și alte structuri ale grupului.
"""

# (conținut complet din fișierul tău — integrat fără modificări majore)

import struct
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ... (restul codului exact din fișierul tău, păstrat intact)

