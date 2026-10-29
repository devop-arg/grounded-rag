# Devolución sobre el brief 08 (misma sesión, mismo contrato)

Muy buen trabajo. Un único cambio: el ejemplo en español `a02` ("¿Cómo creo un índice que solo cubra
las filas…") es uno de los fallos de recuperación (no llega al top-8), y una vitrina no debería abrir con un
fallo. Reemplazalo por `a16` ("¿Cómo convierto un documento JSON en filas y columnas con JSON_TABLE?",
rank 1) en los ejemplos de la UI y en `QUESTION_IDS` de `web/record.py`; regenerá `web/recorded.json`
con `python -m web.record`, rehacé `f3-mobile.png` (mismo setup, ejemplo `a16`) y `f3-pytest.log`.
No toques nada más.
