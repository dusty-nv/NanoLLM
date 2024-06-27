
function alertColor(level) {
  if( level == 'success' ) return 'limegreen';
  else if( level == 'error' ) return 'orange';
  else if( level == 'warning' ) return 'orange';
  else return 'rgb(200,200,200)';
}

function toTimeString(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', hour12: true, minute: '2-digit', second: '2-digit' }).slice(0,-3); // remove AM/PM
}
  
function onHideAlerts() {
  $('#alert_window').fadeOut('slow', function() {
    $(`#alert_messages pre`).remove();
  });
}

function removeAlert(id) {
  $(`#alert_messages #alert-${id}`).remove();
}

function addAlert(alert) {
  // supress other messages from the same category that may still be showing
  if( alert['category'].length > 0 )
    $(`#alert_messages .alert-category-${alert['category']}`).remove();
    
  // add a new element containing the alert message, and show the window if needed
  $('#alert_messages').append(`<pre id="alert-${alert['id']}" class="align-middle m-0 alert-category-${alert['category']}" style="color: ${alertColor(alert['level'])}; font-size: 115%;">[${alert['time']}] ${alert['message']}\n</pre>`);
  $('#alert_window:hidden').fadeIn('fast');

  // automatically remove the messages (if this is the last message, hide the window too)
  if( alert['timeout'] > 0 ) {
    const alert_id = alert['id'];
    setTimeout(function() {
      if( $('#alert_messages pre').length > 1 ) {
        $(`#alert_messages #alert-${alert_id}`).remove();
        console.log('removing alerts due to timeout');
        console.log(`#alert_messages #alert-${alert_id}`);
      }
      else if( $(`#alert_messages #alert-${alert['id']}`).length > 0 ) {
        onHideAlerts();
      }
    }, alert['timeout']);
  }
}
