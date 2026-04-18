# Start all services
up: 
  docker compose up -d 

up-build:
  docker compose up -d --build

rebuild:
  docker compose build --no-cache
  just up
    
# Start and watch for Dockerfile/dep changes
watch: 
  docker compose up --watch

# Stop all services
down: 
  docker compose down

clean:
  docker compose down --rmi local  -v --remove-orphans

# Tail logs for all services
logs: 
  docker compose logs -f

# Tail logs for a specific service: just log kafka
log service: 
  docker compose logs -f {{service}}

# Run model training (uses train-once profile)
train: 
  docker compose --profile manual run --rm train-once

# Build PDF report and presentation from LaTeX sources
report:
  cd reports && lualatex -interaction=nonstopmode -halt-on-error streamcab_report.tex
  cd reports && lualatex -interaction=nonstopmode -halt-on-error streamcab_report.tex

presentation:
  cd reports && lualatex -interaction=nonstopmode -halt-on-error streamcab_presentation.tex
  cd reports && lualatex -interaction=nonstopmode -halt-on-error streamcab_presentation.tex

# Clean LaTeX build artifacts
report-clean: 
  cd reports && rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz

# Show running container status
ps: 
  docker compose ps
