# Devolución sobre el brief 05 (misma sesión, mismo contrato y reglas)

Buen trabajo. Un cambio acotado en `app/limits.py` / `app/main.py`:

1. **Sal vacía**: con `GROUNDED_RAG_IP_SALT` vacía, `sha256("" + ip)` es reversible por fuerza bruta (el
   espacio IPv4 es chico). Si la sal configurada está vacía, usar una sal aleatoria por proceso
   (`secrets.token_hex(32)`), sin imprimirla ni loguearla; con sal configurada, usarla tal cual.
   Documentarlo en `.env.example` (comentario) y en el README (una línea).
2. **Limpieza de claves**: en `RateLimiter.allow`, si tras purgar la ventana el deque quedó vacío y la
   request se rechaza o no, no dejar crecer el dict indefinidamente: eliminar claves con deque vacío
   (por ejemplo, barrido barato cada N llamadas). Test.

Tests nuevos para ambos puntos; regenerá `f2-pytest.log` (todo en verde). No toques nada más.
Entregable: diff resumido y salida de pytest.
