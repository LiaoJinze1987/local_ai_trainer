package com.liao.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            AiChatScreen()
        }
    }
}

@Composable
fun AiChatScreen() {
    val aiClient = remember { AiClient() }
    val scope = rememberCoroutineScope()

    var prompt by remember { mutableStateOf("") }
    var response by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("正在检查模型状态...") }
    var loading by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        try {
            val loaded = withContext(Dispatchers.IO) {
                aiClient.checkStatus()
            }

            status = if (loaded) {
                "模型已加载"
            } else {
                "模型尚未加载"
            }
        } catch (e: Exception) {
            status = "AI 服务连接失败：${e.message}"
        }
    }

    MaterialTheme {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text(
                text = "AI 对话",
                style = MaterialTheme.typography.headlineMedium
            )

            Text(
                text = status,
                modifier = Modifier.padding(top = 8.dp)
            )

            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 16.dp),
                label = {
                    Text("请输入问题")
                },
                minLines = 3
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 12.dp),
                horizontalArrangement = Arrangement.End
            ) {
                Button(
                    onClick = {
                        scope.launch {
                            loading = true

                            try {
                                response = withContext(Dispatchers.IO) {
                                    aiClient.chat(
                                        prompt = prompt,
                                        maxNewTokens = 200,
                                        temperature = 0.7
                                    )
                                }
                            } catch (e: Exception) {
                                response = "请求失败：${e.message}"
                            } finally {
                                loading = false
                            }
                        }
                    },
                    enabled = !loading && prompt.isNotBlank()
                ) {
                    Text("发送")
                }
            }

            if (loading) {
                CircularProgressIndicator(
                    modifier = Modifier.padding(top = 16.dp)
                )
            }

            Text(
                text = "AI 回复",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 20.dp)
            )

            Text(
                text = response,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp)
                    .verticalScroll(rememberScrollState())
            )
        }
    }
}