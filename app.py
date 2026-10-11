from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify
)

import sqlite3
import os
import base64
import io

from datetime import datetime
from functools import wraps

import numpy as np
from PIL import Image


# ============================================================
# TENSORFLOW
# ============================================================

try:
    import tensorflow as tf

except Exception as e:

    tf = None

    print("TensorFlow error:", e)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = "smartwaste-ai-secret-key-2026"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE = os.path.join(
    BASE_DIR,
    "smartwaste.db"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "waste_model.h5"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)


os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# CNN CLASS NAMES
# ============================================================

CLASS_NAMES = [

    "cardboard",

    "glass",

    "metal",

    "paper",

    "plastic",

    "trash"

]


# ============================================================
# DRY / WET WASTE
# ============================================================

WASTE_CATEGORY = {

    "cardboard":
        "Dry Waste",

    "glass":
        "Dry Waste",

    "metal":
        "Dry Waste",

    "paper":
        "Dry Waste",

    "plastic":
        "Dry Waste",

    "trash":
        "Wet Waste"

}


# ============================================================
# RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {

    "cardboard":
        "Flatten cardboard and keep it clean and dry. Put it in the paper/cardboard recycling stream.",

    "glass":
        "Keep glass separate from general waste and place bottles or containers in the appropriate glass recycling collection.",

    "metal":
        "Clean metal cans and containers and place them in the appropriate metal recycling collection.",

    "paper":
        "Keep paper clean and dry and place it in the paper recycling stream.",

    "plastic":
        "Empty and clean the plastic container and place it in the appropriate plastic recycling stream.",

    "trash":
        "Dispose of this waste in the appropriate waste bin."
}


# ============================================================
# MODEL
# ============================================================

cnn_model = None


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# GET TABLE COLUMNS
# ============================================================

def get_columns(
    conn,
    table
):

    rows = conn.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    return [
        row["name"]
        for row in rows
    ]


# ============================================================
# ADD COLUMN IF MISSING
# ============================================================

def add_column_if_missing(
    conn,
    table,
    column,
    definition
):

    columns = get_columns(
        conn,
        table
    )

    if column not in columns:

        conn.execute(
            f"""
            ALTER TABLE {table}
            ADD COLUMN {column} {definition}
            """
        )

        print(
            f"Added column {table}.{column}"
        )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    print(
        "Initializing database..."
    )

    conn = get_db_connection()


    # ========================================================
    # USERS TABLE
    # ========================================================

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            is_admin INTEGER DEFAULT 0,

            created_at TEXT

        )
        """
    )


    # ========================================================
    # CLASSIFICATIONS TABLE
    # ========================================================

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS classifications (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER,

            prediction TEXT,

            waste_type TEXT,

            waste_category TEXT,

            confidence REAL DEFAULT 0,

            recyclable INTEGER DEFAULT 0,

            recommendation TEXT,

            image_path TEXT,

            created_at TEXT

        )
        """
    )


    # ========================================================
    # REPAIR OLD USERS TABLE
    # ========================================================

    add_column_if_missing(
        conn,
        "users",
        "name",
        "TEXT DEFAULT 'User'"
    )

    add_column_if_missing(
        conn,
        "users",
        "email",
        "TEXT"
    )

    add_column_if_missing(
        conn,
        "users",
        "password",
        "TEXT DEFAULT ''"
    )

    add_column_if_missing(
        conn,
        "users",
        "is_admin",
        "INTEGER DEFAULT 0"
    )

    add_column_if_missing(
        conn,
        "users",
        "created_at",
        "TEXT"
    )


    # ========================================================
    # REPAIR OLD CLASSIFICATION TABLE
    # ========================================================

    add_column_if_missing(
        conn,
        "classifications",
        "user_id",
        "INTEGER"
    )

    add_column_if_missing(
        conn,
        "classifications",
        "prediction",
        "TEXT"
    )

    add_column_if_missing(
        conn,
        "classifications",
        "waste_type",
        "TEXT"
    )

    # NEW COLUMN
    add_column_if_missing(
        conn,
        "classifications",
        "waste_category",
        "TEXT"
    )

    add_column_if_missing(
        conn,
        "classifications",
        "confidence",
        "REAL DEFAULT 0"
    )

    add_column_if_missing(
        conn,
        "classifications",
        "recyclable",
        "INTEGER DEFAULT 0"
    )

    add_column_if_missing(
        conn,
        "classifications",
        "recommendation",
        "TEXT"
    )

    add_column_if_missing(
        conn,
        "classifications",
        "image_path",
        "TEXT"
    )

    add_column_if_missing(
        conn,
        "classifications",
        "created_at",
        "TEXT"
    )


    # ========================================================
    # SYNCHRONIZE OLD RECORDS
    # ========================================================

    conn.execute(
        """
        UPDATE classifications

        SET prediction = waste_type

        WHERE
            (prediction IS NULL
             OR prediction = '')

        AND waste_type IS NOT NULL
        AND waste_type != ''
        """
    )


    conn.execute(
        """
        UPDATE classifications

        SET waste_type = prediction

        WHERE
            (waste_type IS NULL
             OR waste_type = '')

        AND prediction IS NOT NULL
        AND prediction != ''
        """
    )


    # ========================================================
    # SET DRY / WET CATEGORY FOR OLD RECORDS
    # ========================================================

    for waste_type, category in WASTE_CATEGORY.items():

        conn.execute(
            """
            UPDATE classifications

            SET waste_category = ?

            WHERE
                (
                    waste_type = ?
                    OR prediction = ?
                )

            AND
                (
                    waste_category IS NULL
                    OR waste_category = ''
                )
            """,
            (
                category,
                waste_type,
                waste_type
            )
        )


    # ========================================================
    # ADMIN USER
    # ========================================================

    admin = conn.execute(
        """
        SELECT id
        FROM users
        WHERE email = ?
        """,
        (
            "admin@smartwaste.com",
        )
    ).fetchone()


    if admin is None:

        conn.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                password,
                is_admin,
                created_at
            )

            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "Administrator",
                "admin@smartwaste.com",
                "admin123",
                1,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

    else:

        conn.execute(
            """
            UPDATE users

            SET is_admin = 1

            WHERE email = ?
            """,
            (
                "admin@smartwaste.com",
            )
        )


    conn.commit()

    conn.close()

    print(
        "Database ready."
    )


# ============================================================
# LOAD CNN MODEL
# ============================================================

def load_cnn_model():

    global cnn_model


    if tf is None:

        print(
            "TensorFlow is not available."
        )

        return False


    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "MODEL NOT FOUND:"
        )

        print(
            MODEL_PATH
        )

        return False


    try:

        cnn_model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print(
            "CNN MODEL LOADED"
        )

        print(
            "Output shape:",
            cnn_model.output_shape
        )

        return True


    except Exception as e:

        print(
            "MODEL LOAD ERROR:",
            e
        )

        cnn_model = None

        return False


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(
    image
):

    image = image.convert(
        "RGB"
    )


    image = image.resize(
        (160, 160)
    )


    image_array = np.array(
        image,
        dtype=np.float32
    )


    image_array = (
        image_array / 255.0
    )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# CLASSIFY IMAGE
# ============================================================

def classify_image(
    image
):

    global cnn_model


    if cnn_model is None:

        if not load_cnn_model():

            return {

                "success": False,

                "error":
                    "CNN model could not be loaded. "
                    "Make sure model/waste_model.h5 exists."

            }


    try:

        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        processed = prepare_image(
            image
        )


        # ----------------------------------------------------
        # CNN PREDICTION
        # ----------------------------------------------------

        predictions = cnn_model.predict(
            processed,
            verbose=0
        )


        predictions = np.asarray(
            predictions
        )


        if predictions.ndim == 2:

            predictions = predictions[0]


        print(
            "CNN predictions:",
            predictions
        )


        # ----------------------------------------------------
        # CHECK SIX OUTPUTS
        # ----------------------------------------------------

        if len(predictions) != 6:

            return {

                "success": False,

                "error":
                    "CNN model must have exactly 6 output classes."

            }


        # ----------------------------------------------------
        # GET PREDICTED CLASS
        # ----------------------------------------------------

        index = int(
            np.argmax(
                predictions
            )
        )


        waste_type = CLASS_NAMES[
            index
        ]


        # ----------------------------------------------------
        # DRY / WET CATEGORY
        # ----------------------------------------------------

        waste_category = WASTE_CATEGORY.get(
            waste_type,
            "Unknown"
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = float(
            predictions[index]
        )


        if confidence <= 1:

            confidence *= 100


        confidence = max(
            0,
            min(
                confidence,
                100
            )
        )


        # ----------------------------------------------------
        # RECYCLABLE
        # ----------------------------------------------------

        recyclable = (
            waste_type != "trash"
        )


        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        recommendation = RECOMMENDATIONS.get(
            waste_type,
            "Dispose of the waste appropriately."
        )


        # ----------------------------------------------------
        # LOW CONFIDENCE WARNING
        # ----------------------------------------------------

        warning = None


        if confidence < 50:

            warning = (
                "AI confidence is low. "
                "Try another clearer image."
            )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {

            "success":
                True,

            "prediction":
                waste_type,

            "waste_category":
                waste_category,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "recyclable":
                recyclable,

            "recommendation":
                recommendation,

            "warning":
                warning

        }


    except Exception as e:

        print(
            "CNN ERROR:",
            e
        )


        return {

            "success":
                False,

            "error":
                "CNN prediction failed: "
                + str(e)

        }


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(
    function
):

    @wraps(function)

    def wrapper(
        *args,
        **kwargs
    ):

        if "user_id" not in session:

            if request.path == "/predict":

                return jsonify({

                    "success":
                        False,

                    "error":
                        "Please login again."

                }), 401


            flash(
                "Please login first.",
                "error"
            )


            return redirect(
                url_for("login")
            )


        return function(
            *args,
            **kwargs
        )


    return wrapper


# ============================================================
# ADMIN REQUIRED
# ============================================================

def admin_required(
    function
):

    @wraps(function)

    def wrapper(
        *args,
        **kwargs
    ):

        if "user_id" not in session:

            return redirect(
                url_for("login")
            )


        if not session.get(
            "is_admin",
            False
        ):

            flash(
                "Admin access required.",
                "error"
            )

            return redirect(
                url_for("dashboard")
            )


        return function(
            *args,
            **kwargs
        )


    return wrapper


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    if "user_id" in session:

        if session.get(
            "is_admin",
            False
        ):

            return redirect(
                url_for(
                    "admin_dashboard"
                )
            )


        return redirect(
            url_for(
                "dashboard"
            )
        )


    return render_template(
        "index.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=[
        "GET",
        "POST"
    ]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()


        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        password = request.form.get(
            "password",
            ""
        )


        if (
            not name
            or not email
            or len(password) < 6
        ):

            flash(
                "Enter valid details. Password must be at least 6 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )


        conn = get_db_connection()


        existing = conn.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (
                email,
            )
        ).fetchone()


        if existing:

            conn.close()

            flash(
                "Email already registered.",
                "error"
            )

            return render_template(
                "register.html"
            )


        try:

            conn.execute(
                """
                INSERT INTO users
                (
                    name,
                    email,
                    password,
                    is_admin,
                    created_at
                )

                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name,
                    email,
                    password,
                    0,
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            )


            conn.commit()

            conn.close()


            flash(
                "Registration successful.",
                "success"
            )


            return redirect(
                url_for("login")
            )


        except Exception as e:

            conn.close()

            flash(
                str(e),
                "error"
            )


    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=[
        "GET",
        "POST"
    ]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        password = request.form.get(
            "password",
            ""
        )


        conn = get_db_connection()


        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (
                email,
            )
        ).fetchone()


        conn.close()


        if (
            user is None
            or user["password"] != password
        ):

            flash(
                "Invalid email or password.",
                "error"
            )

            return render_template(
                "login.html"
            )


        session.clear()


        session["user_id"] = (
            user["id"]
        )

        session["name"] = (
            user["name"]
        )

        session["email"] = (
            user["email"]
        )

        session["is_admin"] = bool(
            user["is_admin"] or 0
        )


        if session["is_admin"]:

            return redirect(
                url_for(
                    "admin_dashboard"
                )
            )


        return redirect(
            url_for(
                "dashboard"
            )
        )


    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("index")
    )


# ============================================================
# DASHBOARD COUNTS
# ============================================================

def get_counts(
    conn,
    user_id=None
):

    if user_id is None:

        where = ""

        params = ()

    else:

        where = (
            "WHERE user_id = ?"
        )

        params = (
            user_id,
        )


    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total = conn.execute(
        f"""
        SELECT COUNT(*) AS n

        FROM classifications

        {where}
        """,
        params
    ).fetchone()["n"]


    # --------------------------------------------------------
    # RECYCLABLE
    # --------------------------------------------------------

    recyclable_where = (
        where
        +
        (
            " AND "
            if where
            else " WHERE "
        )
        +
        "recyclable = 1"
    )


    recyclable = conn.execute(
        f"""
        SELECT COUNT(*) AS n

        FROM classifications

        {recyclable_where}
        """,
        params
    ).fetchone()["n"]


    non_recyclable = (
        total - recyclable
    )


    # --------------------------------------------------------
    # WASTE COUNTS
    # --------------------------------------------------------

    counts = {

        "cardboard": 0,

        "glass": 0,

        "metal": 0,

        "paper": 0,

        "plastic": 0,

        "trash": 0

    }


    rows = conn.execute(
        f"""
        SELECT

            COALESCE(
                NULLIF(
                    waste_type,
                    ''
                ),
                prediction
            ) AS wt,

            COUNT(*) AS n

        FROM classifications

        {where}

        GROUP BY

            COALESCE(
                NULLIF(
                    waste_type,
                    ''
                ),
                prediction
            )
        """,
        params
    ).fetchall()


    for row in rows:

        if row["wt"] in counts:

            counts[
                row["wt"]
            ] = row["n"]


    return (
        total,
        recyclable,
        non_recyclable,
        counts
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    conn = get_db_connection()


    (
        total,
        recyclable,
        non_recyclable,
        waste_counts
    ) = get_counts(
        conn,
        session["user_id"]
    )


    recent = conn.execute(
        """
        SELECT *,

        COALESCE(
            NULLIF(
                waste_type,
                ''
            ),
            prediction
        ) AS display_type

        FROM classifications

        WHERE user_id = ?

        ORDER BY id DESC

        LIMIT 10
        """,
        (
            session["user_id"],
        )
    ).fetchall()


    conn.close()


    return render_template(

        "dashboard.html",

        total=total,

        recyclable=recyclable,

        non_recyclable=non_recyclable,

        waste_counts=waste_counts,

        recent=recent

    )


# ============================================================
# CAMERA
# ============================================================

@app.route("/camera")
@login_required
def camera():

    return render_template(
        "camera.html"
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "user_id" not in session:

        return jsonify({

            "success":
                False,

            "error":
                "Please login again."

        }), 401


    conn = None


    try:

        image = None


        # ====================================================
        # CAMERA IMAGE
        # ====================================================

        if request.is_json:

            data = request.get_json(
                silent=True
            )


            if not data:

                return jsonify({

                    "success":
                        False,

                    "error":
                        "No JSON data received."

                }), 400


            image_data = data.get(
                "image"
            )


            if not image_data:

                return jsonify({

                    "success":
                        False,

                    "error":
                        "No camera image received."

                }), 400


            if "," in image_data:

                image_data = (
                    image_data.split(
                        ",",
                        1
                    )[1]
                )


            try:

                image_bytes = (
                    base64.b64decode(
                        image_data
                    )
                )


                image = Image.open(
                    io.BytesIO(
                        image_bytes
                    )
                ).convert(
                    "RGB"
                )


            except Exception as e:

                return jsonify({

                    "success":
                        False,

                    "error":
                        "Could not read camera image: "
                        + str(e)

                }), 400


        # ====================================================
        # UPLOADED IMAGE
        # ====================================================

        else:

            uploaded_file = (
                request.files.get(
                    "file"
                )
            )


            if (
                uploaded_file is None
                or uploaded_file.filename == ""
            ):

                return jsonify({

                    "success":
                        False,

                    "error":
                        "No image file received."

                }), 400


            try:

                image = Image.open(
                    uploaded_file.stream
                ).convert(
                    "RGB"
                )


            except Exception as e:

                return jsonify({

                    "success":
                        False,

                    "error":
                        "Invalid image: "
                        + str(e)

                }), 400


        # ====================================================
        # CNN
        # ====================================================

        result = classify_image(
            image
        )


        if not result.get(
            "success"
        ):

            return jsonify(
                result
            ), 500


        # ====================================================
        # RESULT VALUES
        # ====================================================

        waste_type = result[
            "prediction"
        ]


        waste_category = result[
            "waste_category"
        ]


        confidence = float(
            result[
                "confidence"
            ]
        )


        recyclable = (
            1
            if result["recyclable"]
            else 0
        )


        recommendation = result[
            "recommendation"
        ]


        # ====================================================
        # SAVE IMAGE
        # ====================================================

        filename = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            + ".jpg"
        )


        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        image.save(
            filepath,
            "JPEG",
            quality=90
        )


        image_url = url_for(
            "static",
            filename=
                "uploads/" + filename
        )


        # ====================================================
        # DATABASE
        # ====================================================

        conn = get_db_connection()


        columns = get_columns(
            conn,
            "classifications"
        )


        values = {}


        if "prediction" in columns:

            values[
                "prediction"
            ] = waste_type


        if "waste_type" in columns:

            values[
                "waste_type"
            ] = waste_type


        # NEW DRY/WET VALUE

        if "waste_category" in columns:

            values[
                "waste_category"
            ] = waste_category


        if "user_id" in columns:

            values[
                "user_id"
            ] = session[
                "user_id"
            ]


        if "confidence" in columns:

            values[
                "confidence"
            ] = confidence


        if "recyclable" in columns:

            values[
                "recyclable"
            ] = recyclable


        if "recommendation" in columns:

            values[
                "recommendation"
            ] = recommendation


        if "image_path" in columns:

            values[
                "image_path"
            ] = image_url


        if "created_at" in columns:

            values[
                "created_at"
            ] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )


        # ----------------------------------------------------
        # OLD CATEGORY COLUMN
        # ----------------------------------------------------

        if "category" in columns:

            values[
                "category"
            ] = waste_type


        # ====================================================
        # INSERT RECORD
        # ====================================================

        column_names = list(
            values.keys()
        )


        placeholders = ",".join(
            ["?"] * len(column_names)
        )


        sql = f"""
            INSERT INTO classifications
            ({",".join(column_names)})

            VALUES
            ({placeholders})
        """


        conn.execute(
            sql,
            tuple(
                values[name]
                for name in column_names
            )
        )


        conn.commit()

        conn.close()

        conn = None


        # ====================================================
        # JSON RESPONSE
        # ====================================================

        return jsonify({

            "success":
                True,

            "prediction":
                waste_type,

            "waste_category":
                waste_category,

            "confidence":
                confidence,

            "recyclable":
                bool(recyclable),

            "recommendation":
                recommendation,

            "warning":
                result.get(
                    "warning"
                ),

            "image":
                image_url

        })


    except Exception as e:

        print(
            "PREDICT ERROR:",
            e
        )


        if conn:

            try:

                conn.close()

            except Exception:

                pass


        return jsonify({

            "success":
                False,

            "error":
                "Server error while processing image: "
                + str(e)

        }), 500


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    conn = get_db_connection()


    rows = conn.execute(
        """
        SELECT *,

        COALESCE(
            NULLIF(
                waste_type,
                ''
            ),
            prediction
        ) AS display_type

        FROM classifications

        WHERE user_id = ?

        ORDER BY id DESC
        """,
        (
            session[
                "user_id"
            ],
        )
    ).fetchall()


    conn.close()


    return render_template(
        "history.html",
        history=rows
    )


# ============================================================
# ANALYTICS
# ============================================================

@app.route("/analytics")
@login_required
def analytics():

    conn = get_db_connection()


    (
        total,
        recyclable,
        non_recyclable,
        waste_counts
    ) = get_counts(
        conn,
        session[
            "user_id"
        ]
    )


    # --------------------------------------------------------
    # DRY / WET COUNTS
    # --------------------------------------------------------

    dry_waste = conn.execute(
        """
        SELECT COUNT(*) AS n

        FROM classifications

        WHERE user_id = ?

        AND waste_category =
            'Dry Waste'
        """,
        (
            session[
                "user_id"
            ],
        )
    ).fetchone()["n"]


    wet_waste = conn.execute(
        """
        SELECT COUNT(*) AS n

        FROM classifications

        WHERE user_id = ?

        AND waste_category =
            'Wet Waste'
        """,
        (
            session[
                "user_id"
            ],
        )
    ).fetchone()["n"]


    conn.close()


    return render_template(

        "analytics.html",

        total=total,

        recyclable=recyclable,

        non_recyclable=non_recyclable,

        waste_counts=waste_counts,

        dry_waste=dry_waste,

        wet_waste=wet_waste

    )


# ============================================================
# RECYCLING CENTERS
# ============================================================

@app.route(
    "/recycling-centers"
)
@login_required
def recycling_centers():

    return render_template(
        "recycling_centers.html"
    )


# ============================================================
# ADMIN
# ============================================================

@app.route("/admin")
@admin_required
def admin():

    return redirect(
        url_for(
            "admin_dashboard"
        )
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route(
    "/admin/dashboard"
)
@admin_required
def admin_dashboard():

    conn = get_db_connection()


    total_users = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM users
        """
    ).fetchone()["n"]


    (
        total_scans,
        recyclable,
        non_recyclable,
        waste_counts
    ) = get_counts(
        conn
    )


    # --------------------------------------------------------
    # DRY / WET TOTALS
    # --------------------------------------------------------

    dry_waste = conn.execute(
        """
        SELECT COUNT(*) AS n

        FROM classifications

        WHERE waste_category =
            'Dry Waste'
        """
    ).fetchone()["n"]


    wet_waste = conn.execute(
        """
        SELECT COUNT(*) AS n

        FROM classifications

        WHERE waste_category =
            'Wet Waste'
        """
    ).fetchone()["n"]


    recent = conn.execute(
        """
        SELECT

            classifications.*,

            COALESCE(
                NULLIF(
                    classifications.waste_type,
                    ''
                ),
                classifications.prediction
            ) AS display_type,

            users.name AS user_name,

            users.email AS user_email

        FROM classifications

        LEFT JOIN users

        ON classifications.user_id =
           users.id

        ORDER BY classifications.id DESC

        LIMIT 10
        """
    ).fetchall()


    conn.close()


    return render_template(

        "admin/admin_dashboard.html",

        total_users=total_users,

        total_scans=total_scans,

        waste_counts=waste_counts,

        dry_waste=dry_waste,

        wet_waste=wet_waste,

        recent=recent

    )


# ============================================================
# ADMIN USERS
# ============================================================

@app.route(
    "/admin/users"
)
@admin_required
def admin_users():

    conn = get_db_connection()


    users = conn.execute(
        """
        SELECT *

        FROM users

        ORDER BY id DESC
        """
    ).fetchall()


    conn.close()


    return render_template(

        "admin/users.html",

        users=users

    )


# ============================================================
# ADMIN CLASSIFICATIONS
# ============================================================

@app.route(
    "/admin/classifications"
)
@admin_required
def admin_classifications():

    conn = get_db_connection()


    classifications = conn.execute(
        """
        SELECT

            classifications.*,

            COALESCE(
                NULLIF(
                    classifications.waste_type,
                    ''
                ),
                classifications.prediction
            ) AS display_type,

            users.name AS user_name,

            users.email AS user_email

        FROM classifications

        LEFT JOIN users

        ON classifications.user_id =
           users.id

        ORDER BY classifications.id DESC
        """
    ).fetchall()


    conn.close()


    return render_template(

        "admin/classifications.html",

        classifications=classifications

    )


# ============================================================
# 404 ERROR
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    if request.path.startswith(
        "/predict"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "API endpoint not found."

        }), 404


    return render_template(
        "index.html"
    ), 404


# ============================================================
# 500 ERROR
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    if request.path.startswith(
        "/predict"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Internal server error."

        }), 500


    return (
        "Internal Server Error",
        500
    )


# ============================================================
# START APPLICATION
# ============================================================

initialize_database()

load_cnn_model()


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("              SMARTWASTE AI")
    print("=" * 60)
    print()

    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Dry Waste:"
    )

    print(
        "Cardboard, Glass, Metal, Paper, Plastic"
    )

    print()

    print(
        "Wet Waste:"
    )

    print(
        "Trash"
    )

    print()

    print("=" * 60)


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )