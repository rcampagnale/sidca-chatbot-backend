# Bienvenida SIDCA por Eventarc

La APP no invoca el envío de bienvenida. El receptor Eventarc se ejecutará en
un servicio Cloud Run separado y privado.

## Arquitectura

| Servicio | Uso | `ENABLE_WELCOME_EVENTARC` | Acceso |
|---|---|---|---|
| `sidca-chatbot-backend` | API actual de la APP y demás funciones | `false` o ausente | conserva su política actual |
| `sidca-welcome-events` | misma imagen/código, sólo receptor Eventarc | `true` | privado, sin `--allow-unauthenticated` |

El servicio principal **no debe hacerse privado** para resolver Eventarc. No se
debe ejecutar `gcloud run services add-iam-policy-binding sidca-chatbot-backend ...`
como parte de esta integración.

El handler acepta únicamente el tipo CloudEvent
`google.cloud.firestore.document.v1.created` y recursos `usuarios/{usuarioId}`.
Después vuelve a leer Firestore, verifica la afiliación positiva y aplica la
reserva/idempotencia del correo.

## Evento y destino

- Proyecto: `sidca-a33f0`
- Firestore Native mode
- Base de datos: `(default)`
- Evento: `google.cloud.firestore.document.v1.created`
- Patrón: `usuarios/{usuarioId}`
- Región: `us-central1`
- Servicio destino: `sidca-welcome-events`
- Ruta: `/internal/events/firestore/usuario-created`

## Feature flag

```text
sidca-chatbot-backend: ENABLE_WELCOME_EVENTARC=false
sidca-welcome-events: ENABLE_WELCOME_EVENTARC=true
```

Con el flag distinto de la cadena exacta `"true"`, el handler responde `404`.
No responde `401` ni `403` y no expone la funcionalidad interna. Con `true`,
procesa el CloudEvent normalmente.

## Comandos propuestos (no ejecutar todavía)

Usar la misma imagen del backend ya construida; estos comandos son sólo una
guía para una etapa posterior:

```powershell
$PROJECT_ID = "sidca-a33f0"
$REGION = "us-central1"
$IMAGE = "REGION-docker.pkg.dev/$PROJECT_ID/REPOSITORY/sidca-chatbot-backend:TAG"
$EVENTARC_SA = "sidca-eventarc@$PROJECT_ID.iam.gserviceaccount.com"

# Servicio privado dedicado; no usar --allow-unauthenticated.
gcloud run deploy sidca-welcome-events `
  --project=$PROJECT_ID `
  --region=$REGION `
  --image=$IMAGE `
  --no-allow-unauthenticated `
  --update-env-vars=ENABLE_WELCOME_EVENTARC=true

# El servicio existente conserva su acceso actual.
gcloud run services update sidca-chatbot-backend `
  --project=$PROJECT_ID `
  --region=$REGION `
  --update-env-vars=ENABLE_WELCOME_EVENTARC=false
```

Antes de ejecutar el trigger, revisar la sintaxis de la versión instalada:

```powershell
gcloud eventarc triggers create --help
```

Trigger propuesto:

```powershell
gcloud eventarc triggers create sidca-usuarios-created `
  --project=$PROJECT_ID `
  --location=$REGION `
  --destination-run-service=sidca-welcome-events `
  --destination-run-region=$REGION `
  --destination-run-path=/internal/events/firestore/usuario-created `
  --event-filters="type=google.cloud.firestore.document.v1.created" `
  --event-filters="database=(default)" `
  --event-filters-path-pattern="document=usuarios/{usuarioId}" `
  --service-account=$EVENTARC_SA
```

## IAM mínimo propuesto

Crear una cuenta dedicada sólo si no existe:

```powershell
gcloud iam service-accounts create sidca-eventarc `
  --project=$PROJECT_ID `
  --display-name="SIDCA Eventarc bienvenida"
```

Permisos runtime:

```powershell
gcloud projects add-iam-policy-binding $PROJECT_ID `
  --member="serviceAccount:$EVENTARC_SA" `
  --role="roles/eventarc.eventReceiver"

gcloud run services add-iam-policy-binding sidca-welcome-events `
  --project=$PROJECT_ID `
  --region=$REGION `
  --member="serviceAccount:$EVENTARC_SA" `
  --role="roles/run.invoker"
```

La identidad administrativa que cree el trigger necesita permisos de Eventarc
y `roles/iam.serviceAccountUser` sobre `$EVENTARC_SA`. El service agent
administrado por Google debe conservar `roles/eventarc.serviceAgent`; no se
deben conceder `roles/editor` ni permisos amplios al runtime.

La única autorización del receptor en producción debe ser Cloud Run IAM más la
service account de Eventarc. No usar API keys ni headers inventados como
mecanismo de autenticación.

## Reafiliación pendiente

La solicitud de reafiliación se guarda en el documento existente
`nuevoAfiliado_counters/{dni}`. Como ese documento se actualiza, el receptor
usa `google.cloud.firestore.document.v1.updated` y vuelve a leer el estado
actual antes de enviar el correo pendiente o el correo de aprobación. La ruta
es `/internal/events/firestore/reafiliacion-pendiente`.

El envío es idempotente por la combinación del DNI y el identificador estable
de `fechaSolicitudReafiliacion`. Las actualizaciones que marcan `procesando`,
`enviado` o `error` generan eventos posteriores, pero se omiten al detectar la
misma solicitud. Una nueva fecha de solicitud permite un nuevo correo para el
mismo DNI.

El mismo receptor distingue los estados `pendiente`, `aprobada` y `rechazada`;
no se necesita un segundo trigger Eventarc. El correo aprobado mantiene sus
propios campos de estado (`correoReafiliacionAprobada*`) y el correo rechazado
usa `correoReafiliacionRechazadaEstado`,
`correoReafiliacionRechazadaProcesandoAt`,
`correoReafiliacionRechazadaEnviadoAt`,
`correoReafiliacionRechazadaUltimoErrorAt`,
`correoReafiliacionRechazadaIdProveedor`,
`correoReafiliacionRechazadaUltimoError` y
`correoReafiliacionRechazadaSolicitudId`. Todos reutilizan la misma solicitud
lógica basada en `fechaSolicitudReafiliacion`, por lo que las actualizaciones
de correo también se omiten de forma idempotente y no generan loops.

Cuando el estado es `rechazada`, el receptor usa la fecha real de
`fechaResolucionReafiliacion` y el motivo real de `observacionResolucion`.
Si no existe motivo, no agrega un bloque vacío ni inventa una causa. El correo
rechazado no incluye teléfonos: sólo muestra los canales oficiales web,
Facebook, YouTube e Instagram.

Comando propuesto (no ejecutar todavía):

```powershell
$PROJECT_ID = "sidca-a33f0"
$EVENTARC_SA = "sidca-eventarc@$PROJECT_ID.iam.gserviceaccount.com"

gcloud eventarc triggers create sidca-reafiliacion-pendiente `
  --project=$PROJECT_ID `
  --location=southamerica-east1 `
  --destination-run-service=sidca-welcome-events `
  --destination-run-region=us-central1 `
  --destination-run-path=/internal/events/firestore/reafiliacion-pendiente `
  --event-filters="type=google.cloud.firestore.document.v1.updated" `
  --event-filters="database=(default)" `
  --event-filters-path-pattern="document=nuevoAfiliado_counters/{dni}" `
  --service-account=$EVENTARC_SA
```

El trigger requiere la misma cuenta dedicada y los permisos mínimos ya
documentados para Eventarc; no se cambia la política IAM del backend principal.
