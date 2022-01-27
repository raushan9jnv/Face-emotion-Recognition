mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = var port = process.env.PORT || 8051;
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
