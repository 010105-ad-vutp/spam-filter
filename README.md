# spam-filter
Сървър, който се използва за класифициране на текст, връщайки вероятността съдържанието да представлява спам

## Предпоставки
Програмата използва вектори от думи, чрез които се извличат определящите елементи от дадения текст. Този набор от вектори е сравнително 
голям (1гб+) и затова трябва да бъде свален отделно преди да се стартира приложението.
Файлът може да бъде намерен тук:

https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

Трябва да бъде поставен в следната директория:
`PROJECT_ROOT_DIR/src/main/resources/`

## Използване

След като сървъра е стартиран, нещо което отнема няколко минути, е възможно да се проверяват съобщения като се изпрати `POST` заявка на 
`http://localhost:8080/spam-filter`, която съдържа текстът, който трябва да бъде проверен. Отговорът от сървъра представлява стойност 
от 0 до 1, където 0 показва, че даденото съобщение за проверка не е класифицирано като спам. а 1 - че съобщението действително е спам

## Пример:
`curl -X POST http://localhost:8080/spam-filter -d "Win a free prize now! Credit card details are required"`
Отговор:
`0.9980637431144714`

## Данни
Данните за обучавне и проверка на мрежата могат да бъдат немерни тук:
https://github.com/codenamewei/nlp-with-use-case/tree/master/1_SpamMailFiltering/src/main/resources/data