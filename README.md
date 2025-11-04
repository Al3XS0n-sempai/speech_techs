# speech_techs
Репозиторий по курсу "Речевые технологии" 2025-2026


Запуск:
```shell
    uv run .\main.py --lang ru_ru --model_name ctc --epochs 10 --batch_size 8 --eval_split validation --top_k 3 --out ckpts_ru
```
Валидация:
```shell
    uv run .\eval.py --lang ru_ru --model_name ctc --eval_split validation
```

Можно запустить обучение сразу с проверкой на валидации:
```shell
    uv run .\main.py --lang ru_ru --model_name ctc --epochs 10 --batch_size 8 --eval_split validation --top_k 3 --out ckpts_ru --eval_after_training
```

Полученные результаты (без обработки текста):
```py
{
  "val_best_wer": 0.07198892478080296,
  "test_wer": 0.07385084117101604,
}
```

Полученные результаты (с заменой латиницы на транслит):
```py
{
  "val_best_wer": 0.06875865251499769,
  "test_wer": 0.07093654788713737,
}
```

Полученные результаты (с заменой латиницы на транслит и вместо цифр слова):
```py
{
  "val_best_wer": 0.06370392840891854,
  "validation_wer": 0.06370392840891854,
}
```

Кроме библиотек из `pyproject.toml` нужно поставить:
```shell
    uv pip install git@github.com:salute-developers/GigaAM.git
```
И для преобразования цифр нужен `num2words` (самый простой варик):
```shell
        uv pip install num2words
```

---


###  `TextNormalizer`

Осуществляет предобработку текста:

* приведение к нижнему регистру
* удаление пунктуации
* сохранение цифр и латиницы
* опционально:

  * `--numbers_mode spell`: цифры → слово (`5000` → `пять тысяч`)
  * `--latin_mode translit`: латиница → русские буковки (`apple` → `эпл`). Вспомнил что такое диграфы :)

---

###  `FleursDataset`

Загружает аудио и текст из `google/fleurs`:

* аудио преобразуется в `torch.Tensor`
* стерео -> моно
* к транскрибациям  примется нормализатор

---

###  `FleursCollate`

Собирает batch:

* ресэмплинг аудио -> частота модели (16 kHz)
* паддинг аудио
* кодирование текста в последовательность индексов (CTC targets)

Ну короч как в 1 ДЗ

---

###  `CTCTrainer`

Отвечает за всё обучение и оценку.
Включает:

* загрузку GigaAM модели
* оптимизатор (`AdamW`)
* mixed-precision (AMP)
* сохранение чекпоинтов:
  * Интересует по сути только `best.pt`
* подсчёт WER (для простоты использую `jiwer.wer`)
* сохранение результатов в `metrics.json`
