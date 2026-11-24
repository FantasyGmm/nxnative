"""Generate iOS embed assembly from a synthetic module, without Xcode or game files."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / 'tools/gen_embed.py'


class GenEmbedTests(unittest.TestCase):
    def test_layout_with_alias_cut_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = root / 'main.patched.bin'
            module.write_bytes(bytes(0x8010))
            layout = ('# PATCH_OFF\t0x8000\n'
                      '# KEY_PAGE_OFF\t0x4000\n'
                      '# KEY_PAGE_SIZE\t0x4000\n'
                      '# TEXT_END\t0x3800\n'
                      '# ALIAS_CUT\t0x100\te001c0d2\t2000c0d2\n')
            fixups = root / 'main.fixups.tsv'
            fixups.write_text(layout)
            asm = root / 'embed.S'

            result = subprocess.run([sys.executable, str(TOOL), str(root), str(asm), 'main'], capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), '-Wl,-segprot,__DMmainW,rw,rw')
            directives = [line.strip() for line in asm.read_text().splitlines() if line.lstrip().startswith(('.section ', '.incbin '))]
            self.assertEqual(directives, [
                '.section __TEXT,__dmmainx0', f'.incbin "{module}", 0x0, 0x4000',
                '.section __DMmainW,__img', f'.incbin "{module}", 0x4000, 0x4000',
                '.section __TEXT,__dmmainx1', f'.incbin "{module}", 0x8000, 0x10',
            ])
            self.assertEqual(fixups.read_text(), layout)


if __name__ == '__main__':
    unittest.main()
