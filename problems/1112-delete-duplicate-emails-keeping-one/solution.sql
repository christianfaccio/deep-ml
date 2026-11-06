-- your query
SELECT MIN(id) as id, email
FROM person
GROUP BY email 
ORDER BY id;