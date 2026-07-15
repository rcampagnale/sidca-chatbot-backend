# Mercado Pago Checkout Pro - SIDCA

Este backend prepara el circuito seguro para la cuota sindical de adherente con Mercado Pago Checkout Pro.

## Variables de entorno

- `MP_ENV`: `test` o `production`.
- `MP_ACCESS_TOKEN_TEST`: access token de prueba.
- `MP_ACCESS_TOKEN`: access token productivo.
- `MP_BACK_URL_SUCCESS`: deep link de retorno aprobado.
- `MP_BACK_URL_PENDING`: deep link de retorno pendiente.
- `MP_BACK_URL_FAILURE`: deep link de retorno rechazado/error.
- `MP_WEBHOOK_URL`: URL pública del webhook. Solo se envía a Mercado Pago si tiene valor.
- `MP_WEBHOOK_SECRET`: secret usado para validar `x-signature`.
- `FIREBASE_PROJECT_ID`: proyecto Firebase/Google Cloud.

No usar `localhost` como back URL de Checkout Pro.

## Configuración Firestore requerida

Documento: `config/cuotaAdherente`

```json
{
  "habilitada": true,
  "periodo": 2026,
  "importe": 50000,
  "moneda": "ARS",
  "concepto": "Cuota sindical de adherente SIDCA 2026",
  "detalle": "Regularización de cuota sindical correspondiente al período 2026",
  "cuotasMaximas": 1
}
```

El backend no confía en importe, concepto, detalle ni nombre enviados desde el frontend. La preferencia se arma con la configuración de Firestore y los datos reales del afiliado.

## Endpoints

### `POST /api/pagos/mercadopago/preference`

Endpoint seguro para crear o reutilizar una preferencia.

Requiere:

- `Authorization: Bearer FIREBASE_ID_TOKEN`
- Body: `{ "dni": "12345678" }`

Valida que el DNI pertenezca al UID autenticado mediante `usuarios_dni/{dni}` o campos UID equivalentes en `usuarios` / `nuevoAfiliado`.

Respuesta:

```json
{
  "ok": true,
  "pagoId": "...",
  "preferenceId": "...",
  "checkoutUrl": "...",
  "ambiente": "test"
}
```

### `POST /api/pagos/mercadopago/webhook`

Recibe notificaciones de Mercado Pago. El backend:

1. valida `x-signature` y `x-request-id`;
2. consulta el pago real en Mercado Pago;
3. valida monto, moneda, `external_reference` y ambiente;
4. registra idempotencia en `pagos_mercadopago/{paymentId}`;
5. actualiza `pagos_adherentes/{pagoId}`;
6. si está aprobado, activa el adherente en `usuarios` y `nuevoAfiliado`;
7. guarda auditoría en `pagos_adherentes/{pagoId}/eventos`.

### `GET /api/pagos/mercadopago/estado/:pagoId`

Consulta segura de un pago propio.

### `GET /api/pagos/mercadopago/mis-pagos`

Lista segura de pagos propios, ordenados del más reciente al más antiguo.

## Colecciones usadas

- `pagos_adherentes`: orden interna SIDCA.
- `pagos_mercadopago`: idempotencia por pago Mercado Pago.
- `pagos_adherentes/{pagoId}/eventos`: auditoría.
- `usuarios`, `nuevoAfiliado`: activación del adherente.
- `usuarios_dni`: vínculo DNI ↔ UID autenticado.

## Comprobante

El comprobante solo se expone para pagos aprobados e incluye la leyenda:

> Comprobante de pago. No válido como factura.
