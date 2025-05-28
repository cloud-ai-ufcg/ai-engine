FROM golang:1.24.2 AS builder

WORKDIR /app

COPY go.mod ./
COPY go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the Go app
RUN go build -o actuator .

# Use a minimal runtime image
FROM debian:bookworm-slim

WORKDIR /root/

# Create .kube directory
RUN mkdir -p /root/.kube

# Copy binary from builder
COPY --from=builder /app/actuator .

# Copy example CSV file if needed (optional)
COPY recommendations_example.csv ./recommendations.csv

# Create volume mount point for kubeconfig
VOLUME ["/root/.kube"]

# Command to run
CMD ["./actuator"]
