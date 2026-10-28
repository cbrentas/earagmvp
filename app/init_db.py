from app.db import initialize_database

if __name__ == '__main__':
    initialize_database()
    print('Created pgvector extension and application tables.')
