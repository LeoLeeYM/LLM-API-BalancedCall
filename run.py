from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=9000,
        debug=app.config.get('DEBUG', False),
        threaded=True
    )