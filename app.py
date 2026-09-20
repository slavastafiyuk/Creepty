import sys

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QHeaderView, QLabel, QMainWindow, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from pipeline.segmenter import DEFAULT_SYSTEM, Scene, split_story


class Gerador(QObject):
    """Runs outside the UI thread so the UI doesn't get blocked."""

    progresso = Signal(str)
    cenas_prontas = Signal(list)
    terminado = Signal(str)
    falhou = Signal(str)

    def __init__(self, historia: str, prompt: str):
        super().__init__()
        self.historia = historia
        self.prompt = prompt

    def correr(self):
        try:
            self.progresso.emit("A dividir a história em cenas...")
            cenas = split_story(self.historia, self.prompt)
            self.cenas_prontas.emit(cenas)
            self.terminado.emit(f"Pronto. {len(cenas)} cenas.")
        except Exception as erro:
            self.falhou.emit(f"Falhou: {erro}")


class Janela(QMainWindow):
    def __init__(self):
        super().__init__()
        # Keep references alive: a local QThread gets collected mid-run.
        self.worker = None
        self.thread = None

        self.setWindowTitle("Creepty")
        self.resize(900, 780)

        self.prompt = QTextEdit()
        self.prompt.setPlainText(DEFAULT_SYSTEM)
        self.prompt.setMaximumHeight(120)

        self.historia = QTextEdit()
        self.historia.setPlaceholderText(
            "Cola aqui o texto que queres transformar em vídeo."
        )

        self.botao = QPushButton("Gerar vídeo")
        self.botao.clicked.connect(self.gerar)

        self.reset = QPushButton("Repor prompt")
        self.reset.clicked.connect(
            lambda: self.prompt.setPlainText(DEFAULT_SYSTEM)
        )

        self.estado = QLabel("")

        self.tabela = QTableWidget(0, 3)
        self.tabela.setHorizontalHeaderLabels(["Texto", "Imagem", "Mood"])
        self.tabela.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tabela.setWordWrap(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addWidget(QLabel("Instruções para o modelo"))
        layout.addWidget(self.prompt)
        layout.addWidget(self.reset)
        layout.addWidget(QLabel("História"))
        layout.addWidget(self.historia)
        layout.addWidget(self.botao)
        layout.addWidget(self.estado)
        layout.addWidget(self.tabela)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def mostrar_cenas(self, cenas: list[Scene]):
        self.tabela.setRowCount(len(cenas))
        for linha, cena in enumerate(cenas):
            for coluna, valor in enumerate(
                [cena.text, cena.image_prompt, cena.mood]
            ):
                self.tabela.setItem(linha, coluna, QTableWidgetItem(valor))
        self.tabela.resizeRowsToContents()

    def gerar(self):
        texto = self.historia.toPlainText().strip()
        prompt = self.prompt.toPlainText().strip()

        if not texto:
            self.estado.setText("Cola uma história para começar.")
            return
        if not prompt:
            self.estado.setText("O prompt não pode ficar vazio.")
            return

        self.botao.setEnabled(False)
        self.tabela.setRowCount(0)

        self.thread = QThread()
        self.worker = Gerador(texto, prompt)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.correr)
        self.worker.progresso.connect(self.estado.setText)
        self.worker.cenas_prontas.connect(self.mostrar_cenas)
        self.worker.terminado.connect(self.estado.setText)
        self.worker.falhou.connect(self.estado.setText)
        self.worker.terminado.connect(self.thread.quit)
        self.worker.falhou.connect(self.thread.quit)
        self.thread.finished.connect(lambda: self.botao.setEnabled(True))

        self.thread.start()


if __name__ == "__main__":
    qt = QApplication(sys.argv)
    janela = Janela()
    janela.show()
    sys.exit(qt.exec())
