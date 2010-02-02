

from openmdao.main.component import Component

class DumbComponent(Component):
    def __init__(self):
        super(DumbComponent, self).__init__()
        self.fnum = 3.14
        self.inum = 2
        self.svar = 'abcdefg'
        self.version = '0.1'

    def execute(self, required_outputs=None):
        self.fnum += 2.0
        self.inum -= 3
        self.svar = self.svar[::-1]

