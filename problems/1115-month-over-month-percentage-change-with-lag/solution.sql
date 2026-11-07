-- your query
SELECT date_trunc('month', sale_date) AS month, SUM(amount) AS total,
ROUND((SUM(amount) - LAG(SUM(AMOUNT)) OVER (ORDER BY date_trunc('month', sale_date))) / LAG(SUM(amount)) OVER (ORDER BY date_trunc('month', sale_date)) * 100, 2) AS pct_change
FROM sales
GROUP BY date_trunc('month', sale_date)
ORDER BY month;