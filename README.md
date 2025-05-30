[![CI](https://github.com/maxGrigorenko/DistributionClassifier/actions/workflows/ci.yml/badge.svg)](https://github.com/maxGrigorenko/DistributionClassifier/actions/workflows/ci.yml)

## Общее описание
Проект посвящен анализу и сравнению статистических распределений, а также созданию классификатора распределений:
- **Нормальное vs Лапласа** в папке `/src/normal_laplace`
- **Экспоненциальное vs Парето** в папке `/src/exp_pareto`

## Описание инфраструктуры

- Язык программирования: `Python 3.11`
- Форматтер: `black`
- Модульное тестирование: `pytest`

## Отчет по экспериментам

Файл [report/report.pdf](https://github.com/maxGrigorenko/DistributionClassifier/blob/main/report/report.pdf)

## Структура репозитория

- `/report` -- папка с технической документацией, отчетом и исходниками отчета
- `/src` -- папка с исходным кодом функций и папками с экспериментами конкретных распределений
- `/src/exp_pareto` -- Exp vs Pareto
- `/src/normal_laplace` -- Normal vs Laplace
- `/tests` -- папка с модульными тестами
