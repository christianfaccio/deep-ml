-- your query
SELECT e.name AS employee  
FROM employees AS e 
JOIN employees m ON e.manager_id = m.id  
WHERE e.salary > m.salary  