{{- define "helm-web-app.name" -}}
{{ .Chart.Name }}
{{- end }}

{{- define "helm-web-app.fullname" -}}
{{ .Release.Name }}-{{ .Chart.Name }}
{{- end }}