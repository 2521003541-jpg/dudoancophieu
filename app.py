
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
