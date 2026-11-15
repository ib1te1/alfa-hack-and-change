# B06: оценка и выбранные веса

Ручная оценка, подбор и проверка API завершены. [Итоговые метрики, происхождение
оценок и ограничения](../reports/b06_final.md). Реализация — [PR #39](https://github.com/ib1te1/alfa-hack-and-change/pull/39),
загрузка весов — [PR #38](https://github.com/ib1te1/alfa-hack-and-change/pull/38).
Дополнительные ответы не требуются. До принятия PR карточки остаются review,
после принятия соответствующих PR — done.

## Протокол

23 фиксированных кандидата сравниваются только на 22 tune-профилях. Выбран
history=0,5; остальные веса сохранены. Оценка на 14 holdout не влияет на выбор.
Настройка улучшила tune, но немного уступила начальным весам на holdout;
после измерения веса не менялись. Сравнение с бейзлайном и unjudgeable показаны
отдельно, статистическая значимость и рост конверсии не заявляются.

Исходные A/B сохранены побайтно. Подтверждённые копии меняют только independent
в соответствии с подтверждённым двухучастниковым процессом. Перенос между
пакетами проверяет весь видимый профиль, историю, карточку и применимость
инструкции. Исходные хеши пакетов не переписываются.

## Воспроизведение

CPU-окружение: requirements-retrieval.txt и requirements-encoder.txt,
локальные веса принятой E5-small. MODEL_PATH задаёт локальную папку модели.
Нужен исходный экспорт data/handoff; в Git он не хранится.

```sh
python -m alfarec.eval.tune_weights select --answers reports/b06/supplement-9ebd99c2/confirmed-A.json reports/b06/supplement-9ebd99c2/confirmed-B.json --holdout-answers reports/b06/holdout-f09a5163/empty-A.json reports/b06/holdout-f09a5163/derived-B.json
python -m alfarec.eval
```

prepare повторно не нужен: эксперимент зафиксирован в artifacts/weight_tuning.
select возвращает 0 только при завершении, 2 при нехватке ответов, 1 при
неверных входах. Файл artifacts/weights.json автоматически читается API при
старте. Итоговые оценки и происхождение связаны хешами с весами и отчётом.

## Сохранённые материалы

- [Исторический пакет](../reports/b06/historical-652a8151/README.md).
- [Дополнение A159/B182](../reports/b06/supplement-9ebd99c2/README.md).
- [Три финальные holdout-пары](../reports/b06/holdout-f09a5163/README.md): две новые
  оценки B и один проверенный перенос; A не назначено ни одной пары.
- [Эксперимент](../artifacts/weight_tuning/experiment.json),
  [выбор](../artifacts/weight_tuning/selection.json),
  [итоговые оценки](../artifacts/weight_tuning/final_grades.json).

Формы в artifacts/weight_tuning сохранены для аудита. Их повторное заполнение
не требуется. Разбор возможных расхождений поддерживают ключи --resolutions
и --holdout-resolutions; фактически расхождений в принятых ответах нет.
