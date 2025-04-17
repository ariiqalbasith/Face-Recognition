import os
import cv2
import numpy as np
import csv
from datetime import datetime
import psycopg2
from psycopg2 import sql
import logging
from threading import Thread

# Konfigurasi logging
logging.basicConfig(filename='attendance.log', level=logging.INFO)


# Koneksi ke PostgreSQL
def connect_db():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="face_attendance",
            user="postgres",
            password="admin",
            connect_timeout=3
        )
        return conn
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL: {e}")
        print(f"Error connecting to database: {e}")
        return None


# Inisialisasi database
def init_db():
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()

            # Buat tabel employees jika belum ada
            cur.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id SERIAL PRIMARY KEY,
                    nip VARCHAR(20),
                    name VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Buat tabel face_data jika belum ada
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_data (
                    id SERIAL PRIMARY KEY,
                    employee_id INTEGER REFERENCES employees(id),
                    image_path VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Buat tabel attendance jika belum ada
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id SERIAL PRIMARY KEY,
                    employee_id INTEGER REFERENCES employees(id),
                    check_in TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    check_out TIMESTAMP,
                    status VARCHAR(20)
                )
            """)

            conn.commit()
            logging.info("Database initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            print(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()


# Fungsi untuk input data karyawan
def input_employee_data():
    print("\n=== INPUT DATA KARYAWAN ===")
    nip = input("Masukkan NIP (kosongkan jika tidak ada): ").strip()
    name = input("Masukkan Nama: ").strip()

    if not name:
        print("Nama tidak boleh kosong!")
        return

    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO employees (nip, name) VALUES (%s, %s) RETURNING id",
                (nip if nip else None, name)
            )
            employee_id = cur.fetchone()[0]
            conn.commit()
            print(f"Data karyawan berhasil disimpan dengan ID: {employee_id}")
            logging.info(f"New employee added: ID={employee_id}, Name={name}")
            capture_face_images(employee_id)
        except Exception as e:
            logging.error(f"Error saving employee data: {e}")
            print(f"Error saving employee data: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()


# Fungsi untuk mengambil gambar wajah
def capture_face_images(employee_id):
    # Pastikan folder dataset ada
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    count = 0
    conn = connect_db()

    print("\nTekan SPASI untuk mengambil gambar. Ambil 20 gambar dari berbagai angle.")
    print("Tekan ESC untuk keluar.")

    while count < 20:
        ret, img = cam.read()
        if not ret:
            print("Gagal mengambil gambar dari kamera")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Tampilkan hitungan di frame
            cv2.putText(img, f"Count: {count}/20", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            k = cv2.waitKey(1) & 0xff
            if k == 32:  # SPASI untuk mengambil gambar
                count += 1
                img_name = f"dataset/User.{employee_id}.{count}.jpg"
                cv2.imwrite(img_name, gray[y:y + h, x:x + w])

                # Simpan path gambar ke database
                if conn:
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            "INSERT INTO face_data (employee_id, image_path) VALUES (%s, %s)",
                            (employee_id, img_name)
                        )
                        conn.commit()
                        print(f"Gambar {count} disimpan: {img_name}")
                    except Exception as e:
                        print(f"Error saving face data: {e}")
                        conn.rollback()
                    finally:
                        if cur: cur.close()

        cv2.imshow('Capture Face - Press SPACE', img)

        k = cv2.waitKey(1) & 0xff
        if k == 27:  # ESC untuk keluar
            break

    cam.release()
    cv2.destroyAllWindows()

    if conn:
        conn.close()

    # Train model setelah mengambil gambar
    train_face_recognizer_async()


# Fungsi untuk melatih model (dijalankan di thread terpisah)
def train_face_recognizer_async():
    print("\nMemulai training model di background...")
    Thread(target=train_face_recognizer, daemon=True).start()


def train_face_recognizer():
    try:
        # Menggunakan cara yang kompatibel dengan OpenCV 4.x
        recognizer = cv2.face.LBPHFaceRecognizer_create() if hasattr(cv2.face,
                                                                     'LBPHFaceRecognizer_create') else cv2.face.createLBPHFaceRecognizer()

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def get_images_and_labels():
            image_paths = []
            conn = connect_db()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT image_path FROM face_data")
                    image_paths = [row[0] for row in cur.fetchall()]
                except Exception as e:
                    print(f"Error fetching image paths: {e}")
                finally:
                    cur.close()
                    conn.close()

            face_samples = []
            ids = []

            for image_path in image_paths:
                if os.path.exists(image_path):
                    try:
                        # Dapatkan ID dari nama file
                        id = int(os.path.split(image_path)[-1].split(".")[1])
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        faces = face_cascade.detectMultiScale(img)

                        for (x, y, w, h) in faces:
                            face_samples.append(img[y:y + h, x:x + w])
                            ids.append(id)
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")

            return face_samples, ids

        print("\nTraining data wajah. Harap tunggu...")
        faces, ids = get_images_and_labels()

        if len(faces) > 0:
            recognizer.train(faces, np.array(ids))

            # Simpan model yang sudah dilatih
            if not os.path.exists('trainner'):
                os.makedirs('trainner')
            recognizer.save('trainner/trainner.yml')
            print(f"\n{len(np.unique(ids))} wajah berhasil dilatih.")
            logging.info(f"Model trained with {len(faces)} samples from {len(np.unique(ids))} people")
        else:
            print("\nTidak ada data wajah untuk dilatih.")
            logging.warning("No face data available for training")
    except Exception as e:
        print(f"Error during training: {e}")
        logging.error(f"Training error: {e}")


# Fungsi untuk presensi
def face_attendance():
    try:
        # Load recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create() if hasattr(cv2.face,
                                                                     'LBPHFaceRecognizer_create') else cv2.face.createLBPHFaceRecognizer()

        if os.path.exists('trainner/trainner.yml'):
            recognizer.read('trainner/trainner.yml')
        else:
            print("Model pengenalan wajah belum dilatih. Silakan input data terlebih dahulu.")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        print("\nArahkan wajah ke kamera. Sistem akan otomatis mendeteksi.")
        print("Tekan SPASI atau 'y' untuk konfirmasi jika wajah benar.")
        print("Tekan ESC untuk keluar.")

        confirmed = False
        last_recognized = None

        while True:
            ret, img = cam.read()
            if not ret:
                print("Gagal membaca kamera")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Jika confidence kurang dari 100 -> wajah dikenali
                if confidence < 100:
                    conn = connect_db()
                    if conn:
                        try:
                            cur = conn.cursor()
                            cur.execute("SELECT id, name FROM employees WHERE id = %s", (id,))
                            result = cur.fetchone()

                            if result:
                                employee_id, name = result
                                confidence_text = f"  {round(100 - confidence)}%"
                                cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                                cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                                # Tampilkan instruksi konfirmasi
                                cv2.putText(img, "Tekan SPASI/y jika benar", (10, 30),
                                            font, 0.8, (0, 255, 0), 2)

                                last_recognized = (employee_id, name)
                        except Exception as e:
                            print(f"Error during recognition: {e}")
                        finally:
                            cur.close()
                            conn.close()
                else:
                    # Wajah tidak dikenali
                    cv2.putText(img, "Tidak Dikenali", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                    last_recognized = None

            cv2.imshow('Presensi Wajah - Konfirmasi dengan SPASI/y', img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:  # ESC untuk keluar
                break
            elif (k == 32 or k == ord('y')) and last_recognized:  # SPASI atau 'y' untuk konfirmasi
                employee_id, name = last_recognized

                # Cek apakah sudah presensi hari ini
                conn = connect_db()
                if conn:
                    try:
                        cur = conn.cursor()
                        today = datetime.now().date()
                        cur.execute(
                            """SELECT id FROM attendance 
                            WHERE employee_id = %s AND DATE(check_in) = %s""",
                            (employee_id, today)
                        )
                        existing = cur.fetchone()

                        if not existing:
                            # Simpan presensi
                            cur.execute(
                                "INSERT INTO attendance (employee_id, status) VALUES (%s, 'Hadir')",
                                (employee_id,)
                            )
                            conn.commit()
                            print(f"Presensi berhasil untuk: {name}")
                            logging.info(f"Attendance recorded for {name} (ID: {employee_id})")

                            # Tampilkan notifikasi sukses
                            cv2.putText(img, "PRESENSI BERHASIL", (50, 50),
                                        font, 1, (0, 255, 0), 2)
                            cv2.imshow('Presensi Wajah - Konfirmasi dengan SPASI/y', img)
                            cv2.waitKey(1000)  # Tampilkan notifikasi selama 1 detik
                        else:
                            print(f"{name} sudah presensi hari ini")
                    except Exception as e:
                        print(f"Error during attendance: {e}")
                        conn.rollback()
                    finally:
                        cur.close()
                        conn.close()

        cam.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in face attendance: {e}")
        logging.error(f"Face attendance error: {e}")


# Fungsi untuk melihat laporan harian
def view_daily_report():
    date_str = input("Masukkan tanggal (YYYY-MM-DD) atau kosongkan untuk hari ini: ").strip()

    try:
        if date_str:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            date = datetime.now().date()
    except ValueError:
        print("Format tanggal tidak valid. Gunakan YYYY-MM-DD")
        return

    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT e.nip, e.name, a.check_in, a.check_out, a.status 
                FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE DATE(a.check_in) = %s
                ORDER BY a.check_in""",
                (date,)
            )

            attendances = cur.fetchall()

            if not attendances:
                print(f"\nTidak ada data presensi untuk tanggal {date}")
                return

            print(f"\nLaporan Presensi Tanggal {date}")
            print("=" * 85)
            print(f"{'NIP':<15} {'Nama':<20} {'Check In':<20} {'Check Out':<20} {'Status':<10}")
            print("-" * 85)

            for att in attendances:
                nip, name, check_in, check_out, status = att
                check_in_str = check_in.strftime("%Y-%m-%d %H:%M:%S") if check_in else "-"
                check_out_str = check_out.strftime("%Y-%m-%d %H:%M:%S") if check_out else "-"
                print(f"{nip or '-':<15} {name:<20} {check_in_str:<20} {check_out_str:<20} {status:<10}")

            # Tanya apakah ingin ekspor ke CSV
            export = input("\nEkspor ke CSV? (y/n): ").lower()
            if export == 'y':
                export_to_csv(attendances, date)
        except Exception as e:
            print(f"Error fetching attendance data: {e}")
            logging.error(f"Report generation error: {e}")
        finally:
            cur.close()
            conn.close()


# Fungsi untuk ekspor ke CSV
def export_to_csv(data, date):
    filename = f"attendance_report_{date}.csv"

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['NIP', 'Nama', 'Check In', 'Check Out', 'Status'])

            for row in data:
                nip, name, check_in, check_out, status = row
                check_in_str = check_in.strftime("%Y-%m-%d %H:%M:%S") if check_in else ""
                check_out_str = check_out.strftime("%Y-%m-%d %H:%M:%S") if check_out else ""
                writer.writerow([
                    nip or "",
                    name,
                    check_in_str,
                    check_out_str,
                    status
                ])

            print(f"Laporan berhasil diekspor ke {filename}")
            logging.info(f"Report exported to {filename}")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        logging.error(f"CSV export error: {e}")


# Menu utama
def main_menu():
    init_db()

    while True:
        print("\n=== SISTEM ABSENSI BERBASIS PENGENALAN WAJAH ===")
        print("1. Input Data Karyawan")
        print("2. Presensi")
        print("3. Lihat Laporan Harian")
        print("4. Keluar")

        choice = input("Pilih menu (1-4): ").strip()

        if choice == '1':
            input_employee_data()
        elif choice == '2':
            face_attendance()
        elif choice == '3':
            view_daily_report()
        elif choice == '4':
            print("Keluar dari program...")
            logging.info("Application closed")
            break
        else:
            print("Pilihan tidak valid. Silakan pilih 1-4.")


if __name__ == "__main__":
    # Buat folder yang diperlukan
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    if not os.path.exists('trainner'):
        os.makedirs('trainner')

    print("Pastikan Anda telah:")
    print("1. Menginstall PostgreSQL dan membuat database 'face_attendance'")
    print("2. Mengubah konfigurasi database di fungsi connect_db()")
    print("3. Menjalankan 'pip install opencv-contrib-python numpy psycopg2-binary'")

    main_menu()