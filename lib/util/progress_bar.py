from abc import ABC

import tqdm.notebook


class ProgressBarProvider(ABC):

    def provide(self, total: int, desc: str) -> tqdm.std.tqdm:
        pass


class NotebookProgressBarProvider(ProgressBarProvider):

    def provide(self, total: int, desc: str = "") -> tqdm.std.tqdm:
        return tqdm.notebook.tqdm(range(total), desc=desc)


NOTEBOOK_PROGRESS_BAR = NotebookProgressBarProvider()
