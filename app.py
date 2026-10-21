from datetime import date, timedelta
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

warnings.filterwarnings("ignore", category=FutureWarning)

# StockLab: phiên bản giao diện và cấu trúc triển khai độc lập
st.set_page_config(
    page_title="StockLab | Phân tích thị trường",
    page_icon="📊",
    layout="wide",
)

st.title("📊 StockLab — Phân tích và dự báo thị trường")
st.write(
    "Khám phá diễn biến giá lịch sử, đánh giá mô hình chuỗi thời gian "
    "và tạo kịch bản giá tham khảo bằng mạng LSTM."
)
st.caption(
    "Ứng dụng phục vụ học tập và nghiên cứu. Kết quả mô hình không bảo đảm "
    "diễn biến thực tế và không phải khuyến nghị đầu tư."
)


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_series(symbol: str, start_day: date, end_day: date) -> pd.DataFrame:
    """Lấy dữ liệu và chuẩn hóa giá đóng cửa đã điều chỉnh."""
    raw = yf.download(
        symbol,
        start=start_day.isoformat(),
        end=(end_day + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Close"])

    if isinstance(raw.columns, pd.MultiIndex):
        close_cols = [
            col for col in raw.columns
            if any(str(level) == "Close" for level in col)
        ]
        if not close_cols:
            return pd.DataFrame(columns=["Close"])
        close = raw[close_cols[0]]
    elif "Close" in raw.columns:
        close = raw["Close"]
    else:
        return pd.DataFrame(columns=["Close"])

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    series = pd.DataFrame(
        {"Close": pd.to_numeric(close, errors="coerce")},
        index=pd.to_datetime(raw.index),
    )
    series = series[~series.index.duplicated(keep="last")]
    series = series.sort_index().dropna()
    series = series[np.isfinite(series["Close"]) & (series["Close"] > 0)]
    return series


def create_training_windows(normalized: np.ndarray, lookback: int):
    """Biến chuỗi giá thành cặp dữ liệu đầu vào và mục tiêu."""
    inputs, targets = [], []
    for position in range(lookback, len(normalized)):
        inputs.append(normalized[position - lookback:position, 0])
        targets.append(normalized[position, 0])

    x = np.asarray(inputs, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    return x, y


def make_lstm(lookback: int) -> Sequential:
    """Khởi tạo mô hình hồi quy chuỗi thời gian."""
    network = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(40, return_sequences=True),
        Dropout(0.2),
        LSTM(24),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    network.compile(optimizer="adam", loss="mse")
    return network


def generate_projection(network, recent_values, scaler, periods: int):
    """Dự báo nhiều bước bằng cách lần lượt đưa dự báo vào cửa sổ đầu vào."""
    sequence = np.asarray(recent_values, dtype=np.float32).reshape(1, -1, 1)
    scaled_forecast = []

    for _ in range(periods):
        prediction = float(network.predict(sequence, verbose=0)[0, 0])
        scaled_forecast.append(prediction)
        next_item = np.asarray([[[prediction]]], dtype=np.float32)
        sequence = np.concatenate([sequence[:, 1:, :], next_item], axis=1)

    return scaler.inverse_transform(
        np.asarray(scaled_forecast, dtype=np.float32).reshape(-1, 1)
    ).ravel()


# ------------------ Tùy chọn của người dùng ------------------
with st.sidebar:
    st.header("Bảng điều khiển")
    symbol = st.text_input(
        "Mã chứng khoán",
        value="VNM.VN",
        help="Ví dụ: VNM.VN, FPT.VN, AAPL, MSFT.",
    ).strip().upper()

    current_day = date.today()
    date_a, date_b = st.columns(2)
    first_day = date_a.date_input(
        "Ngày bắt đầu", value=date(2018, 1, 1), max_value=current_day
    )
    last_day = date_b.date_input(
        "Ngày kết thúc", value=current_day, max_value=current_day
    )

    st.divider()
    st.subheader("Tham số mô hình")
    lookback = st.select_slider(
        "Độ dài lịch sử đầu vào", options=[30, 60, 90], value=60
    )
    epoch_limit = st.slider("Số epoch tối đa", 5, 50, 20, step=5)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)

    forecast_length = st.number_input(
        "Độ dài dự báo", min_value=1, max_value=10, value=1, step=1
    )
    forecast_unit = st.selectbox("Đơn vị thời gian", ["Phiên", "Tháng", "Năm"])
    forecast_count = (
        int(forecast_length)
        if forecast_unit == "Phiên"
        else int(forecast_length * 22)
        if forecast_unit == "Tháng"
        else int(forecast_length * 252)
    )
    st.caption(f"Quy đổi: khoảng {forecast_count:,} phiên giao dịch.")
    run_model = st.button(
        "Bắt đầu phân tích", type="primary", use_container_width=True
    )


if not run_model:
    st.info(
        "Chọn mã chứng khoán và khoảng thời gian trong bảng điều khiển, "
        "sau đó nhấn **Bắt đầu phân tích**."
    )
    st.markdown(
        """
        **Các chức năng chính**
        - Tải và hiển thị dữ liệu giá lịch sử.
        - Huấn luyện mạng LSTM với tập validation.
        - Đánh giá kết quả trên phần dữ liệu kiểm thử theo thời gian.
        - Dự báo nhiều phiên, xem biểu đồ và xuất bảng CSV.
        """
    )
else:
    if not symbol:
        st.error("Bạn chưa nhập mã chứng khoán.")
        st.stop()
    if first_day >= last_day:
        st.error("Ngày bắt đầu cần sớm hơn ngày kết thúc.")
        st.stop()

    try:
        # Cố định seed để tăng khả năng tái lập kết quả.
        np.random.seed(21)
        tf.keras.utils.set_random_seed(21)

        with st.spinner(f"Đang tải dữ liệu cho {symbol}..."):
            history_df = get_market_series(symbol, first_day, last_day)

        if history_df.empty:
            st.error(
                "Không tìm thấy dữ liệu phù hợp. Hãy kiểm tra mã chứng khoán "
                "hoặc thử chọn khoảng ngày khác."
            )
            st.stop()

        raw_prices = history_df["Close"].to_numpy(dtype=np.float32).reshape(-1, 1)
        required_count = max(100, lookback + 30)
        if len(raw_prices) < required_count:
            st.error(
                f"Khoảng dữ liệu có {len(raw_prices)} phiên; nên có ít nhất "
                f"{required_count} phiên với cấu hình hiện tại."
            )
            st.stop()

        # Tách dữ liệu theo trình tự thời gian; scaler chỉ fit trên tập train.
        boundary = int(len(raw_prices) * 0.8)
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(raw_prices[:boundary])
        all_scaled = scaler.transform(raw_prices)

        x_pool, y_pool = create_training_windows(train_scaled, lookback)
        if len(x_pool) < 20:
            st.error("Số mẫu huấn luyện quá ít. Hãy giảm độ dài lịch sử đầu vào.")
            st.stop()

        validation_size = max(1, int(len(x_pool) * 0.1))
        if len(x_pool) - validation_size < 2:
            st.error("Không đủ mẫu sau khi tách tập validation.")
            st.stop()

        x_train, y_train = x_pool[:-validation_size], y_pool[:-validation_size]
        x_valid, y_valid = x_pool[-validation_size:], y_pool[-validation_size:]

        # Dùng các phiên trước mốc test làm ngữ cảnh, không xáo trộn thời gian.
        x_test, y_test = create_training_windows(
            all_scaled[boundary - lookback:], lookback
        )
        test_dates = history_df.index[boundary:]
        if len(x_test) != len(test_dates):
            st.error("Dữ liệu kiểm thử không khớp với mốc thời gian.")
            st.stop()

        tf.keras.backend.clear_session()
        model = make_lstm(lookback)
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        with st.spinner("Đang huấn luyện mạng LSTM..."):
            fit_result = model.fit(
                x_train,
                y_train,
                validation_data=(x_valid, y_valid),
                epochs=epoch_limit,
                batch_size=batch_size,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0,
            )

        with st.spinner("Đang đánh giá và tạo dự báo..."):
            predicted_scaled = model.predict(x_test, verbose=0)
            actual_prices = scaler.inverse_transform(y_test).ravel()
            test_predictions = scaler.inverse_transform(predicted_scaled).ravel()
            future_prices = generate_projection(
                model, all_scaled[-lookback:, 0], scaler, forecast_count
            )

        rmse = float(np.sqrt(mean_squared_error(actual_prices, test_predictions)))
        mae = float(mean_absolute_error(actual_prices, test_predictions))
        r2 = float(r2_score(actual_prices, test_predictions)) if len(actual_prices) > 1 else float("nan")

        latest_price = float(raw_prices[-1, 0])
        next_price = float(future_prices[0])
        predicted_change = (next_price / latest_price - 1) * 100

        st.success(f"Đã phân tích {len(history_df):,} phiên của {symbol}.")
        st.subheader("Tóm tắt kết quả")
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Giá đóng cửa gần nhất", f"{latest_price:,.2f}")
        metric2.metric(
            "Dự báo phiên tiếp theo", f"{next_price:,.2f}", f"{predicted_change:+.2f}%"
        )
        metric3.metric("RMSE kiểm thử", f"{rmse:,.3f}")
        metric4.metric("MAE kiểm thử", f"{mae:,.3f}")
        st.caption(
            f"R² của tập kiểm thử: {r2:.4f}. Các chỉ số này phản ánh dữ liệu kiểm thử "
            "trong quá khứ, không đảm bảo hiệu quả dự báo tương lai."
        )

        projected_dates = pd.bdate_range(
            start=history_df.index[-1] + pd.offsets.BDay(1),
            periods=forecast_count,
        )
        forecast_df = pd.DataFrame({
            "Ngày dự kiến": projected_dates,
            "Phiên dự báo": np.arange(1, forecast_count + 1),
            "Giá ước tính": future_prices,
        })

        chart_tab, training_tab, forecast_tab, prices_tab = st.tabs([
            "Biểu đồ giá", "Quá trình học", "Bảng dự báo", "Dữ liệu lịch sử"
        ])

        with chart_tab:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(test_dates, actual_prices, label="Giá thực tế (test)", linewidth=1.7)
            ax.plot(test_dates, test_predictions, label="Dự báo trên test", linewidth=1.5)
            ax.plot(projected_dates, future_prices, label="Dự báo tương lai", linewidth=2)
            ax.axvline(
                history_df.index[-1], linestyle="--", alpha=0.7,
                label="Ngày cuối dữ liệu",
            )
            ax.set_title(f"Giá đóng cửa và kịch bản dự báo — {symbol}")
            ax.set_xlabel("Thời gian")
            ax.set_ylabel("Giá đóng cửa đã điều chỉnh")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.info(
                "Ngày tương lai là ngày làm việc ước tính, chưa tính chính xác lịch nghỉ "
                "của từng sàn. Dự báo xa có thể tích lũy sai số."
            )

        with training_tab:
            loss_fig, loss_ax = plt.subplots(figsize=(9, 4))
            loss_ax.plot(fit_result.history["loss"], label="Train loss")
            loss_ax.plot(fit_result.history["val_loss"], label="Validation loss")
            loss_ax.set_title("Sai số trong quá trình huấn luyện")
            loss_ax.set_xlabel("Epoch")
            loss_ax.set_ylabel("MSE")
            loss_ax.grid(alpha=0.25)
            loss_ax.legend()
            loss_fig.tight_layout()
            st.pyplot(loss_fig)
            plt.close(loss_fig)
            st.write(f"Mô hình đã chạy {len(fit_result.history['loss'])} epoch.")

        with forecast_tab:
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Tải kết quả CSV",
                data=forecast_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{symbol.lower().replace('.', '_')}_projection.csv",
                mime="text/csv",
            )

        with prices_tab:
            st.line_chart(history_df["Close"])
            st.dataframe(
                history_df.tail(30).sort_index(ascending=False),
                use_container_width=True,
            )

    except Exception as exc:
        st.error("Ứng dụng không thể hoàn thành phân tích.")
        st.exception(exc)
