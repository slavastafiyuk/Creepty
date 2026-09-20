# Creepty

Desktop app para transformar histórias de horror em vídeos curtos.
Cola-se o texto na janela e o pipeline trata do resto.

## Stack

- Python 3.11+
- PySide6 (Qt) para a interface nativa
- Ollama + Gemma 4 E4B para segmentar a história em cenas
- ComfyUI + SDXL para geração de imagem
- Wan 2.2 TI2V-5B (fp8) para os clips de vídeo
- Kokoro-82M para a narração
- moviepy para a montagem final

## Estado

- [x] Janela com caixa de texto e botão de geração
- [x] Pipeline a correr em thread separada para não bloquear a UI
- [x] Prompt editável na UI
- [x] Segmentação da história em cenas com LLM local
- [ ] Ligação ao ComfyUI
- [ ] Narração
- [ ] Montagem com moviepy
- [ ] Publicação

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Instalar o Ollama a partir de ollama.com e puxar o modelo:

```bash
winget install --id Ollama.Ollama
ollama pull gemma4:e4b
pip install ollama pydantic
```

## Correr

```bash
python app.py
```

## Empacotar

```bash
pyinstaller --noconsole --onefile app.py
```
O executavel fica em dist/

## Estrutura

```bash
creepty/
  app.py                janela e ponto de entrada
  pipeline/
    __init__.py
    segmenter.py        divide a história em cenas via Ollama
  requirements.txt
```

## Notas

- Correr sempre a partir da raiz (python app.py), senão o pacote pipeline não é encontrado
- Thread tem de ficar guardado numa referência da janela (self.thread), senão o garbage collector destrói-o a meio da execução
- O Pipeline liga-se no método correr da classe Gerador
- A forma da resposta do modelo vem do schema Pydantic em segmenter.py, não do prompt. Campos novos exigem mexer na classe Scene
- Histórias acima de ~500 palavras podem truncar o JSON. A solução é partir o texto em blocos e chamar o split_story por bloco
- Com 8GB de VRAM, o Ollama e o ComfyUI competem pela mesma memória. OLLAMA_KEEP_ALIVE=0 descarrega o modelo logo após a resposta